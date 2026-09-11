"""Check lossless city filtering, raw CSV strings, and unverified expert provenance."""
import gzip
import json
from scripts.export_handoff_supplements import city_subset, export_regional, export_expert


def test_city_filter_preserves_layers_times_and_units():
    value={'cycle':'2026-01-01T12:00:00Z','sources':[{'cities':{'北京':{'temperature_c':{'925':1.25,'850':-2.5},'wind':{'200':{'u_ms':None}}},'南京':{'temperature_c':{'925':10}}},'available_at':'legacy-estimate'}]}
    result=city_subset(value,{'北京'})
    assert result['sources'][0]['cities']=={'北京':value['sources'][0]['cities']['北京']}
    assert result['sources'][0]['available_at']=='legacy-estimate'
    assert result['cycle']==value['cycle']
    assert '南京' in value['sources'][0]['cities']


def test_regional_export_preserves_station_rows_and_does_not_infer_publication(tmp_path):
    source=tmp_path/'keti';d=source/'cmaq/hour';d.mkdir(parents=True)
    (d/'cmaq_d02_2025121720.csv').write_text('datadate,cityname,stationcode,pm25_1h\n2025-12-18 00:00:00,北京市,A,bad\n2025-12-18 00:00:00,南京市,B,0\n',encoding='utf-8')
    req=tmp_path/'request.json';req.write_text(json.dumps({'cases':[{'city':'北京','issue_date':'2025-12-18'}]}))
    out=tmp_path/'out';export_regional(req,source,out)
    with gzip.open(out/'regional_city_rows.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
    assert len(rows)==1
    assert rows[0]['row']['pm25_1h']=='bad'
    assert rows[0]['row']['stationcode']=='A'
    assert rows[0]['verified_available_at'] is None
    assert rows[0]['cycle_bjt']=='2025-12-17T20:00:00+08:00'


def test_verified_filename_does_not_become_verified_method_or_publication(tmp_path):
    d=tmp_path/'corpus/解压_20260807对话校验原文-test/inner';d.mkdir(parents=True)
    (d/'2025年12月2日_verified.json').write_text(json.dumps({'segments':[{'original_text':'原文','text':'整理文本'}]},ensure_ascii=False))
    out=tmp_path/'expert';export_expert(tmp_path/'corpus',out)
    with gzip.open(out/'expert_text_and_distillations.jsonl.gz','rt') as f:r=json.loads(next(f))
    assert r['meeting_date_claim']=='2025-12-02'
    assert r['published_at'] is None
    assert not r['source_and_date_independently_verified']
    assert r['not_a_verified_method_card']
    assert r['data']['segments'][0]['original_text']=='原文'
