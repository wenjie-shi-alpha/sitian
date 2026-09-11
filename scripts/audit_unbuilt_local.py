#!/usr/bin/env python3
"""Read-only source audit of originally unbuilt Sitian cases; writes new evidence only."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,sqlite3,sys
from collections import Counter,OrderedDict,defaultdict
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import netCDF4

TYPES={'PM2.5':20,'PM10':20,'SO2':20,'NO2':20,'CO':20,'O3_8h':14}
MAP={'PM2.5':'pm25','PM10':'pm10','O3_8h':'o3_8h','SO2':'so2','NO2':'no2','CO':'co'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def identity(path):
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'mtime_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()}

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def cache_store(store):
    slabs=OrderedDict()
    def cached_series(ds,var,ref_utc,lat,lon):
        k=(ds.filepath(),var.name,ref_utc)
        if k not in slabs:
            tv=store._pick(ds,'forecast_reference_time','time')
            times=netCDF4.num2date(tv[:],tv.units)
            ti=next((i for i,t in enumerate(times) if (t.year,t.month,t.day,t.hour)==(ref_utc.year,ref_utc.month,ref_utc.day,ref_utc.hour)),None)
            if ti is None:raise KeyError(f'ref {ref_utc} not in file')
            pv=store._pick(ds,'forecast_period','leadtime_hour','step')
            leads=np.asarray(pv[:],dtype=float)
            if 'seconds' in getattr(pv,'units',''):leads=leads/3600
            arr=var[:,ti] if var.dimensions[0]=='forecast_period' else var[ti]
            if arr.ndim==4:arr=arr[:,0]
            slabs[k]=(leads,arr,np.asarray(ds['latitude'][:]),np.asarray(ds['longitude'][:]))
            if len(slabs)>36:slabs.popitem(last=False)
        leads,arr,lats,lons=slabs[k]
        if not min(lats)<=lat<=max(lats) or not min(lons)<=lon<=max(lons):raise ValueError('city_outside_grid')
        li=int(np.abs(lats-lat).argmin());lj=int(np.abs(lons-lon).argmin())
        return {int(lh):float(arr[i,li,lj]) for i,lh in enumerate(leads) if not np.ma.is_masked(arr[i,li,lj]) and math.isfinite(float(arr[i,li,lj]))}
    store._series=cached_series
    return store

def run(root,out,unbuilt):
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root/'src'))
    builder=load_module(root/'scripts/build_national_case.py','sitian_original_builder')
    eps=json.loads(unbuilt.read_text())
    original_eps=[]
    for split in ('train','val','test'):
        for ep in json.loads((root/f'data/interim/episodes_{split}.json').read_text()):
            if not (root/f'cases/national/{split}/{ep["city"]}_{ep["issue_date"]}/truth.json').exists():
                original_eps.append(dict(ep,split=split))
    key=lambda e:(e['split'],e['city'],e['issue_date'])
    assert set(map(key,eps))==set(map(key,original_eps)), 'Handoff mismatch with local original episodes'
    issues=sorted({e['issue_date'] for e in eps})
    byday=defaultdict(set)
    for e in eps:
        issue=date.fromisoformat(e['issue_date'])
        for offset in range(-3,6):byday[str(issue+timedelta(days=offset))].add(e['city'])
    identities={}
    def record(p):
        p=Path(p)
        if p.is_file() and str(p) not in identities:identities[str(p)]=identity(p)
    for p in [unbuilt,root/'data/aq_daily.sqlite',root/'scripts/build_national_case.py',root/'scripts/build_daily_all.py',root/'scripts/fetch_cams.py',root/'scripts/audit_national_data.py']:
        record(p)
    for directory in ('data/interim','data/verl'):
        for p in (root/directory).glob('*'):
            if p.is_file() and p.suffix in {'.json','.parquet'}:record(p)
    # CSV rows are keyed by actual hour. Invalid/non-finite values do not count.
    cells={}; csv_profiles=[]
    for ds,cities in sorted(byday.items()):
        p=builder.RAW_OBS/ds[:4]/('china_cities_'+ds.replace('-','')+'.csv')
        profile={'date':ds,'path':str(p),'exists':p.is_file(),'cities_requested':len(cities)}
        if not p.is_file():csv_profiles.append(profile);continue
        record(p); data={c:defaultdict(dict) for c in cities}; rows=Counter(); duplicates=[]; bad=[]
        with p.open(encoding='utf-8-sig',newline='') as f:
            r=csv.reader(f); header=next(r); idx={c:header.index(c) for c in cities if c in header}
            profile['missing_city_columns']=sorted(cities-idx.keys())
            for row in r:
                if len(row)<3:bad.append('short_row');continue
                pollutant=row[2]
                if pollutant not in set(TYPES)|{'O3'}:continue
                try:hour=int(row[1])
                except ValueError:bad.append('bad_hour');continue
                if not 0<=hour<=23 or row[0]!=ds.replace('-',''):bad.append('bad_date_or_hour');continue
                rows[pollutant]+=1
                for city,ci in idx.items():
                    raw=row[ci].strip() if ci<len(row) else ''
                    if hour in data[city][pollutant]:duplicates.append([city,pollutant,hour])
                    value=None
                    if raw:
                        try:
                            v=float(raw)
                            if math.isfinite(v) and v>=0:value=v
                            else:bad.append('nonfinite_or_negative')
                        except ValueError:bad.append('nonnumeric')
                    data[city][pollutant][hour]=value
        profile.update(row_counts=dict(rows),duplicate_city_pollutant_hours=len(duplicates),bad_values=dict(Counter(bad)))
        cells[ds]=data;csv_profiles.append(profile)
    dump(out/'observation_file_inventory.json',csv_profiles)
    print('CSV inventory complete:',len(csv_profiles),'files; missing',sum(not x['exists'] for x in csv_profiles),flush=True)
    store=builder.CamsStore()
    profiles=[]; needed={(date.fromisoformat(d)-timedelta(days=1)).isoformat() for d in issues}
    for kind in ('sfc','o3'):
        paths=sorted({store._index[kind][d] for d in needed if d in store._index[kind]})
        for p in paths:
            record(p); prof={'path':str(p),'kind':kind,'bytes':p.stat().st_size}
            try:
                with netCDF4.Dataset(p) as ds:
                    tv=store._pick(ds,'forecast_reference_time','time')
                    times=netCDF4.num2date(tv[:],tv.units)
                    pv=store._pick(ds,'forecast_period','leadtime_hour','step')
                    prof.update(open_ok=True,reference_times=[t.isoformat() for t in times],leads=np.asarray(pv[:],dtype=float).tolist(),lead_units=getattr(pv,'units',''),variables={n:{'dimensions':list(v.dimensions),'units':getattr(v,'units','')} for n,v in ds.variables.items()},latitude_range=[float(np.min(ds['latitude'][:])),float(np.max(ds['latitude'][:]))],longitude_range=[float(np.min(ds['longitude'][:])),float(np.max(ds['longitude'][:]))])
            except Exception as ex:prof.update(open_ok=False,error=repr(ex))
            profiles.append(prof)
    dump(out/'cams_file_inventory.json',profiles)
    # Preserve the builder's extraction semantics; cache already-read ref-time
    # slabs only, avoiding repeated disk decompression for cities sharing a date.
    cache_store(store)
    db=sqlite3.connect(f'file:{builder.DB}?mode=ro',uri=True)
    findings=[]
    for n,ep in enumerate(sorted(eps,key=lambda e:(e['issue_date'],e['city']))):
        city=ep['city']; issue=date.fromisoformat(ep['issue_date']); ref=str(issue-timedelta(days=1))
        r=dict(ep);r['case_id']=f'{city}_{issue}';r['cams_reference_date']=ref
        r['cams_paths']={kind:str(store._index[kind].get(ref,'')) for kind in ('sfc','o3')}
        try:
            guidance,diag=store.extract(city,issue)
            json.dumps([guidance,diag],allow_nan=False)
            r['cams_status']='ok';r['cams_guidance_days']=len(guidance['cams']['daily_pm25']);r['cams_o3_days']=len(guidance['cams'].get('daily_o3max',{}))
        except Exception as ex:r['cams_status']='failed';r['cams_error']=repr(ex)
        obs=builder.read_obs_hours(city,issue)
        r['original_input_pm25_rows']=len(obs.get('PM2.5',{}).get('times',[]))
        r['input_valid_counts']={}
        for pol in builder.OBS_TYPES:
            count=0
            for offset in (-3,-2,-1,0):
                day=str(issue+timedelta(days=offset))
                count+=sum(v is not None for h,v in cells.get(day,{}).get(city,{}).get(pol,{}).items() if offset<0 or h<=7)
            r['input_valid_counts'][pol]=count
        r['input_missing_files']=[str(issue+timedelta(days=i)) for i in (-3,-2,-1,0) if str(issue+timedelta(days=i)) not in cells]
        failures=[]; truth_values={}; dbdiff=[]
        for offset in range(1,6):
            ds=str(issue+timedelta(days=offset)); truth_values[ds]={}
            dbrow=db.execute('SELECT pm25,pm10,o3_8h,so2,no2,co,n_hours FROM daily WHERE city=? AND date=?',(city,ds)).fetchone()
            dbvals=dict(zip(('pm25','pm10','o3_8h','so2','no2','co','n_hours'),dbrow)) if dbrow else {}
            for pol,minimum in TYPES.items():
                vals=[v for v in cells.get(ds,{}).get(city,{}).get(pol,{}).values() if v is not None]
                valid=len(vals)>=minimum or (pol=='O3_8h' and vals and max(vals)>160)
                if not valid:failures.append({'date':ds,'pollutant':pol,'valid_hours':len(vals),'minimum':minimum,'file_missing':ds not in cells})
                value=round(max(vals) if pol=='O3_8h' else sum(vals)/len(vals),1) if valid else None
                truth_values[ds][MAP[pol]]=value
                if value!=dbvals.get(MAP[pol]):dbdiff.append({'date':ds,'pollutant':pol,'raw_value':value,'db_value':dbvals.get(MAP[pol])})
        r['raw_truth_failures']=failures;r['db_raw_discrepancies']=dbdiff
        r['existing_db_truth_valid']=builder.read_truth(db,city,issue) is not None
        r['original_builder_result_now']=('cams_missing' if r['cams_status']!='ok' else 'obs_insufficient' if r['original_input_pm25_rows']<48 else 'truth_incomplete' if not r['existing_db_truth_valid'] else 'ok')
        blockers=[]
        if r['cams_status']!='ok':blockers.append('cams_failure')
        if r['input_valid_counts']['PM2.5']<48:blockers.append('input_observations_below_48_valid_hours')
        if failures:blockers.append('raw_truth_invalid')
        r['current_blockers']=blockers;r['can_rebuild_with_raw_valid_truth']=not blockers
        r['historical_group']='CAMS-missing cohort' if issue.year==2025 else 'observation-insufficient cohort'
        findings.append(r)
        if (n+1)%100==0:print('Cases checked:',n+1,dict(Counter(x['original_builder_result_now'] for x in findings)),flush=True)
    db.close()
    for ds in store._cache.values():ds.close()
    dump(out/'case_findings.json',findings)
    with (out/'case_findings.csv').open('w',encoding='utf-8-sig',newline='') as f:
        names=['case_id','city','issue_date','split','historical_group','cams_status','original_input_pm25_rows','input_pm25_valid_hours','original_builder_result_now','raw_truth_failure_count','db_discrepancy_count','can_rebuild_with_raw_valid_truth','current_blockers']
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for r in findings:w.writerow({k:(r['input_valid_counts']['PM2.5'] if k=='input_pm25_valid_hours' else len(r['raw_truth_failures']) if k=='raw_truth_failure_count' else len(r['db_raw_discrepancies']) if k=='db_discrepancy_count' else ';'.join(r[k]) if k=='current_blockers' else r.get(k)) for k in names})
    dump(out/'source_identities.json',list(identities.values()))
    summary={'checked_cases':len(findings),'local_unbuilt_matches_handoff':True,'issue_days':len(issues),'raw_obs_days':len(byday),'missing_obs_files':[r['date'] for r in csv_profiles if not r['exists']],'cases_by_month':dict(Counter(r['issue_date'][:7] for r in findings)),'cams_status':dict(Counter(r['cams_status'] for r in findings)),'original_builder_results_now':dict(Counter(r['original_builder_result_now'] for r in findings)),'raw_valid_rebuildable':sum(r['can_rebuild_with_raw_valid_truth'] for r in findings),'blocked_reasons':dict(Counter(b for r in findings for b in r['current_blockers'])),'cases_with_stale_db_values':sum(bool(r['db_raw_discrepancies']) for r in findings),'invalid_truth_city_days':len({(r['city'],f['date']) for r in findings for f in r['raw_truth_failures']}),'historical_groups':{g:{'n':len(rs),'raw_valid_rebuildable':sum(r['can_rebuild_with_raw_valid_truth'] for r in rs),'original_builder_results_now':dict(Counter(r['original_builder_result_now'] for r in rs))} for g in {r['historical_group'] for r in findings} for rs in [[r for r in findings if r['historical_group']==g]]},'method':'Local files only; no raw writes, no synthesis, no download, distinct finite hour counts; existing project validity thresholds unchanged.'}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--unbuilt',type=Path,required=True);a=ap.parse_args();run(a.root.resolve(),a.out.resolve(),a.unbuilt.resolve())
