import pandas as pd, numpy as np, warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
warnings.filterwarnings('ignore')

LE_MAPS = {
    'object_class':       {'air_object':0,'land_object':1,'sea_object':2},
    'trajectory_type':    {'ballistic':0,'diving':1,'linear':2},
    'sensor_type':        {'camera':0,'radar':1,'thermal':2},
    'lighting_condition': {'day':0,'dusk':1,'night':2},
    'object_type_label':  {'air_like':0,'land_like':1,'sea_like':2},
    'trajectory_label':   {'ballistic':0,'diving':1,'linear':2},
    'sea_state':          {'calm':0,'moderate':1,'rough':2},
    'terrain_type':       {'forest':0,'mountain':1,'plain':2,'urban':3},
}
THREAT_LABEL  = {0:'HIGH',1:'LOW',2:'MEDIUM'}
THREAT_COLOR  = {'HIGH':'#C0392B','MEDIUM':'#B7770D','LOW':'#1A7A3C'}
THREAT_PASTEL = {'HIGH':'#FADADD','MEDIUM':'#FEF3CD','LOW':'#D4EDDA'}
THREAT_BORDER = {'HIGH':'#E74C3C','MEDIUM':'#F0A500','LOW':'#28A745'}
THREAT_ADVICE = {
    'HIGH':   'Immediate action required — intercept protocol',
    'MEDIUM': 'Elevated alert — maintain close observation',
    'LOW':    'No immediate action — continue standard monitoring',
}

def load_and_train(csv_path):
    df = pd.read_csv(csv_path)
    df_clean = df.copy()
    for c in ['sea_state','terrain_type']:
        df[c] = df[c].fillna(df[c].mode()[0])
        df_clean[c] = df_clean[c].fillna(df_clean[c].mode()[0])
    df['wave_height'] = df['wave_height'].fillna(df['wave_height'].median())
    df_clean['wave_height'] = df_clean['wave_height'].fillna(df_clean['wave_height'].median())
    drop = [c for c in ['object_id','timestamp','frame_id'] if c in df.columns]
    df.drop(columns=drop,inplace=True); df.drop_duplicates(inplace=True)
    df_clean.drop(columns=[c for c in drop if c in df_clean.columns],inplace=True)
    df_clean.drop_duplicates(inplace=True)
    for col,mp in LE_MAPS.items():
        if col in df.columns: df[col]=df[col].map(mp).fillna(0).astype(int)
    le_t = LabelEncoder()
    df['threat_enc'] = le_t.fit_transform(df['threat_level'])
    df['speed_alt_ratio'] = (df['velocity']/(df['altitude']+1)).round(4)
    df['bbox_area'] = (df['bbox_xmax']-df['bbox_xmin'])*(df['bbox_ymax']-df['bbox_ymin'])
    mv = df['visibility_range'].max()
    df['env_difficulty'] = (df['fog_density']*0.33+df['rain_intensity']*0.33+(1-df['visibility_range']/mv)*0.33).round(4)
    X = df.drop(columns=['threat_level','threat_enc'])
    y = df['threat_enc']
    feature_cols = list(X.columns)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = StandardScaler()
    Xtr=scaler.fit_transform(X_train); Xte=scaler.transform(X_test)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200,max_depth=5,random_state=42,n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(max_depth=6,random_state=42),
        'KNN':           KNeighborsClassifier(n_neighbors=7,weights='distance'),
    }
    results = {}
    for name,m in models.items():
        m.fit(Xtr,y_train); yp=m.predict(Xte)
        e={'model':m,'accuracy':accuracy_score(y_test,yp),'f1':f1_score(y_test,yp,average='weighted'),
           'cm':confusion_matrix(y_test,yp),'report':classification_report(y_test,yp,target_names=['HIGH','LOW','MEDIUM'])}
        if hasattr(m,'feature_importances_'): e['importances']=m.feature_importances_
        results[name]=e
    return results,scaler,feature_cols,df_clean

def predict_single(user_input,model,scaler,feature_cols):
    row={c:0.0 for c in feature_cols}
    domain=user_input.get('object_class','air_object')
    row['object_class']=float(LE_MAPS['object_class'].get(domain,0))
    row['object_type_label']=float(LE_MAPS['object_type_label'].get(domain.replace('object','like'),0))
    traj=user_input.get('trajectory_type','linear')
    row['trajectory_type']=float(LE_MAPS['trajectory_type'].get(traj,2))
    row['trajectory_label']=float(LE_MAPS['trajectory_label'].get(traj,2))
    row['sensor_type']=float(LE_MAPS['sensor_type'].get(user_input.get('sensor_type','radar'),1))
    row['lighting_condition']=float(LE_MAPS['lighting_condition'].get(user_input.get('lighting_condition','day'),0))
    row['sea_state']=0.0; row['terrain_type']=2.0
    for k in ['velocity','altitude','trajectory_angle','confidence_score','radar_range',
              'thermal_signature','doppler_velocity','fog_density','rain_intensity',
              'visibility_range','obstacle_density','climb_rate']:
        if k in row and k in user_input: row[k]=float(user_input[k])
    vel=row.get('velocity',500); alt=row.get('altitude',1000)
    row['speed_alt_ratio']=round(vel/(alt+1),4); row['bbox_area']=0.0
    row['env_difficulty']=round(row.get('fog_density',0)*0.33+row.get('rain_intensity',0)*0.33+(1-min(row.get('visibility_range',500),1000)/1000)*0.33,4)
    fv=np.array([[row[c] for c in feature_cols]])
    fv_s=scaler.transform(fv)
    pred=model.predict(fv_s)[0]
    proba=model.predict_proba(fv_s)[0].tolist() if hasattr(model,'predict_proba') else [0.0,0.0,0.0]
    if not hasattr(model,'predict_proba'): proba[int(pred)]=1.0
    return THREAT_LABEL.get(int(pred),'UNKNOWN'),proba