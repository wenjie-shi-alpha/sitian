#!/usr/bin/env python3
"""Validate native exports and derive per-case quality counts without rebuilding cases."""
from __future__ import annotations
import argparse,collections,gzip,json,math,random,sqlite3
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import netCDF4
if __package__:
    from .export_native_city_data import identity
else:
    from export_native_city_data import identity

TYPES={'PM2.5':20,'PM10':20,'SO2':20,'NO2':20,'CO':20,'O3_8h':14}
INPUT=('PM2.5','PM10','O3','SO2','NO2','CO')

def dump(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=1,allow_nan=False)+'\n')

def validate_record(r):
    cycle=datetime.fromisoformat(r['cycle']);assert cycle.hour==12 and cycle.utcoffset()==timedelta(0)
    assert cycle.date()==date.fromisoformat(r['issue_date'])-timedelta(days=1)
    assert all(datetime.fromisoformat(t)==cycle+timedelta(hours=h) for t,h in zip(r['valid_times'],r['lead_hours']))
    assert r['lead_hours']==sorted(set(r['lead_hours']))
    array=np.array(r['values'],dtype=object);axis=r['dimensions'].index(r['lead_dimension'])
    assert array.shape[axis]==len(r['lead_hours'])
    assert sum(v is None for v in array.flat)==r['missing_values']
    assert r['available_at'] is None
    for dim in r['dimensions']:
        coord=r['coordinates'].get(dim)
        if coord is not None:assert len(coord['values'])==array.shape[r['dimensions'].index(dim)]


def direct_source_check(r,source):
    with netCDF4.Dataset(source) as ds:
        var=ds.variables[r['variable']];cycle=datetime.fromisoformat(r['cycle'])
        ref=next(ds.variables[k] for k in ('forecast_reference_time','time') if k in ds.variables)
        dates=netCDF4.num2date(ref[:],ref.units);ti=next(i for i,t in enumerate(dates) if (t.year,t.month,t.day,t.hour)==(cycle.year,cycle.month,cycle.day,cycle.hour))
        y=next(ds.variables[k] for k in ('latitude','lat') if k in ds.variables);x=next(ds.variables[k] for k in ('longitude','lon') if k in ds.variables)
        yi=next(i for i,v in enumerate(y[:]) if float(v)==r['selected_grid']['lat']);xi=next(i for i,v in enumerate(x[:]) if float(v)==r['selected_grid']['lon'])
        fixed={ref.dimensions[0]:ti,y.dimensions[0]:yi,x.dimensions[0]:xi}
        data=np.ma.asarray(var[tuple(fixed.get(d,slice(None)) for d in var.dimensions)],dtype=float).filled(np.nan)
        actual=np.asarray(r['values'],dtype=float)
        assert np.array_equal(data,actual,equal_nan=True),(r['city'],r['issue_date'],r['variable'])
        assert getattr(var,'units',None)==r['unit']


def run(request,batch,out,root):
    out.mkdir(parents=True,exist_ok=False);req=json.loads(request.read_text());manifest=json.loads((batch/'manifest.json').read_text());expected={(r['city'],r['issue_date']) for r in req['cases']}
    sources={r['sha256']:r for r in manifest['sources']};coverage=collections.Counter();seen=collections.defaultdict(set);samples=[];rng=random.Random(20260912);levels=collections.Counter();overlaps=collections.defaultdict(list)
    n=0
    with gzip.open(batch/'cams_native.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);validate_record(r);n+=1
            key=(r['city'],r['issue_date']);seen[key].add(r['canonical_variable'])
            d0=date.fromisoformat(r['issue_date']);days=[0]*5
            for t in r['valid_times']:
                offset=(datetime.fromisoformat(t).astimezone(timezone(timedelta(hours=8))).date()-d0).days
                if 1<=offset<=5:days[offset-1]+=1
            coverage[(r['canonical_variable'],tuple(days))]+=1
            if r['canonical_variable']=='go3':levels[json.dumps(r['coordinates'].get('model_level'),sort_keys=True)]+=1
            if r['issue_date']=='2025-12-19':overlaps[(r['city'],r['canonical_variable'])].append(r)
            if len(samples)<60:samples.append(r)
            else:
                i=rng.randrange(n)
                if i<60:samples[i]=r
    assert set(seen)==expected
    required={'pm2p5','pm10','u10','v10','t2m','d2m','blh','tp','tcc','go3'}
    assert all(v==required for v in seen.values())
    for r in samples:direct_source_check(r,r.get('source_path',sources[r['source_sha256']]['path']))
    overlap_checks=[]
    for (city,var),rs in overlaps.items():
        if len(rs)<2:continue
        a,b=rs;shared=sorted(set(a['lead_hours'])&set(b['lead_hours']));aa=np.asarray(a['values'],float);bb=np.asarray(b['values'],float)
        left=np.take(aa,[a['lead_hours'].index(h) for h in shared],axis=a['dimensions'].index(a['lead_dimension']));right=np.take(bb,[b['lead_hours'].index(h) for h in shared],axis=b['dimensions'].index(b['lead_dimension']))
        overlap_checks.append({'city':city,'variable':var,'source_variants':len(rs),'common_leads':len(shared),'same_unit_and_grid':a['unit']==b['unit'] and a['selected_grid']==b['selected_grid'],'identical_on_common_leads':np.array_equal(left,right,equal_nan=True)})
    dump(out/'native_validation.json',{'records':n,'covered_cases':len(seen),'all_ten_variables_present':True,'shape_time_mask_unit_checks_passed':True,'direct_raw_checks_passed':len(samples),'coverage_by_variable':[{'variable':k[0],'D1_D2_D3_D4_D5_samples':list(k[1]),'records':v} for k,v in sorted(coverage.items())],'o3_level_metadata':dict(levels),'overlapping_source_checks':overlap_checks,'failures_in_export_manifest':manifest['failures']})
    print('Native validated',n,'records',len(seen),'cases',flush=True)
    # Compact hour bitmasks preserve actual distinct-hour completeness per city/day.
    stats={};duplicate_rows=0;rowkeys=set();bad=collections.Counter();obs_sources={}
    with gzip.open(batch/'observations_native.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            try:d=datetime.strptime(r['date'],'%Y%m%d').date().isoformat();h=int(r['hour']);assert 0<=h<24
            except (TypeError,ValueError,AssertionError):bad['invalid_date_or_hour']+=1;continue
            k=(d,h,r['type']);duplicate_rows+=k in rowkeys;rowkeys.add(k);obs_sources[d]=r['source_sha256']
            for city,v in r['city_values'].items():
                key=(city,d,r['type']);s=stats.setdefault(key,[0,0,None]);s[1]|=1<<h
                try:value=float(v)
                except (ValueError,TypeError):continue
                if math.isfinite(value) and value>=0:s[0]|=1<<h;s[2]=value if s[2] is None else max(value,s[2])
    built={(p.parent.name,p.name) for p in (root/'cases/national').glob('*/*') if (p/'truth.json').is_file()}
    valid=set()
    for split in ('train','val','test'):
        valid.update(Path(x).name for x in json.loads((root/f'data/interim/valid_cases_{split}.json').read_text()))
    requested_unbuilt={(r['city'],r['issue_date']) for r in req['unbuilt_cases']};rows=[];counts=collections.Counter()
    for c in req['cases']:
        city=c['city'];issue=date.fromisoformat(c['issue_date']);truth_failures=[];input72={};input80={};raw80=0
        for pol in INPUT:
            n72=0;n80=0
            for offset in (-3,-2,-1,0):
                s=stats.get((city,str(issue+timedelta(days=offset)),pol),[0,0,None]);mask=s[0];present=s[1]
                if offset==0:mask&=255;present&=255
                n80+=mask.bit_count()
                if pol=='PM2.5':raw80+=present.bit_count()
                if offset==-3:mask&=~255
                n72+=mask.bit_count()
            input72[pol]=n72;input80[pol]=n80
        for offset in range(1,6):
            day=str(issue+timedelta(days=offset))
            for pol,minimum in TYPES.items():
                s=stats.get((city,day,pol),[0,0,None]);number=s[0].bit_count();exception=pol=='O3_8h' and s[2] is not None and s[2]>160
                if number<minimum and not exception:truth_failures.append({'date':day,'pollutant':pol,'valid_hours':number,'required':minimum,'source_sha256':obs_sources.get(day)})
        unbuilt=(city,str(issue)) in requested_unbuilt;caseid=f'{city}_{issue}';wasbuilt=any(x[1]==caseid for x in built)
        status='truth_insufficient' if truth_failures else 'raw_truth_valid'
        row={'city':city,'issue_date':str(issue),'original_unbuilt':unbuilt,'previously_built':wasbuilt,'previously_quarantined':wasbuilt and caseid not in valid,'input_finite_hours_72h':input72,'input_finite_hours_builder_window_80h':input80,'original_builder_pm25_timestamp_count':raw80,'truth_failures':truth_failures,'truth_status':status,'original_builder_input_under_48':raw80<48,'cams_native_available':(city,str(issue)) in seen,'training_ready':False,'publication_verified':False}
        row['current_base_build_blockers']=(["input_observations_under_48"] if raw80<48 else [])+(["raw_truth_invalid"] if truth_failures else [])
        rows.append(row);counts[status]+=1
    with gzip.open(out/'all_case_quality.jsonl.gz','wt',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    missing=[r for r in rows if r['original_unbuilt']];assert len(missing)==752
    dump(out/'unbuilt_case_findings.json',missing)
    quarantined=[r for r in rows if r['previously_quarantined']]
    dump(out/'quarantined_case_recheck.json',quarantined)
    summary={'requested_cases':len(rows),'unbuilt_cases':len(missing),'unbuilt_current_blockers':dict(collections.Counter(';'.join(r['current_base_build_blockers']) or 'none_at_base_case_level' for r in missing)),'all_case_truth_status':dict(counts),'previously_quarantined_cases':len(quarantined),'quarantined_still_raw_truth_invalid':sum(bool(r['truth_failures']) for r in quarantined),'duplicate_observation_hour_type_rows':duplicate_rows,'invalid_observation_rows':dict(bad),'input_windows':'72h ending issue 07:00, plus separately recorded legacy builder 80h window; no counts replaced or thresholds reduced','request':identity(request),'extractor':identity(Path(__file__))}
    dump(out/'quality_summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    # Full re-hash of the compact source family, not the much larger GRIB archive.
    changed=[]
    for s in manifest['sources']:
        if identity(Path(s['path']))['sha256']!=s['sha256']:changed.append(s['path'])
    assert not changed
    dump(out/'source_preservation.json',{'raw_source_files_checked':len(manifest['sources']),'changed':changed,'passed':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--batch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--project-root',type=Path,required=True);a=p.parse_args();run(a.request,a.batch,a.out,a.project_root)
