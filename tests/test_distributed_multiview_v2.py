"""Synthetic fixtures only: no malware or future-test model tuning."""
import csv
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

import numpy as np
import pytest

from src.data.multiview_contract import load_manifest, load_mapping, output_paths
from scripts.build_temporal_manifest import main as build_main
from scripts.generate_multiview_dataset_v2 import parse_args, run
from scripts.merge_multiview_dataset import build_parser as merge_parser, merge
from scripts.audit_multiview_dataset import build_parser as audit_parser, audit
from src.data.temporal_multiview_dataset import TemporalMultiViewDataset


def csv_write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def fixture(tmp_path):
    mapping = tmp_path / 'eligible.csv'
    csv_write(mapping, [{'family': f'fam{i}', 'class_id': i, 'n_files': 51, 'n_months': 3}
                        for i in range(51)])
    rng = np.random.default_rng(2026)
    records = [
        ('a'*64, 'train', '2019-10-12T00:00:00+00:00', 'fam0', bytes(range(256))*4 + b'Z'),
        ('b'*64, 'validation', '2020-03-02T00:00:00+00:00', 'fam1', rng.integers(0,256,12800,dtype=np.uint8).tobytes()),
        ('c'*64, 'test', '2020-07-02T00:00:00+00:00', 'fam0', rng.integers(0,256,16385,dtype=np.uint8).tobytes()),
        ('d'*64, 'train', '2019-11-11T00:00:00+00:00', 'fam1', bytes([0])*257),
    ]
    manifest = tmp_path / 'manifest.csv'
    csv_write(manifest,[{'sha':sha, 'family':fam, 'class_id':int(fam[3:]),'split':split,
                         'timestamp':stamp,'month':stamp[:7], 'file_size':'', 'zip_crc32':''}
                        for sha,split,stamp,fam,data in records])
    folder = tmp_path / 'altered'; folder.mkdir()
    archive = tmp_path / 'binaries.zip'
    with ZipFile(archive, 'w', compression=ZIP_STORED) as zf:
        for sha,split,stamp,fam,data in records:
            (folder / (sha+'.exe')).write_bytes(data)
            zf.writestr('altered/'+sha+'.exe',data)
    return {'mapping': mapping, 'manifest': manifest, 'archive':archive,'folder':folder,
            'records':records,'tmp':tmp_path}


def gen(f, modality, output, source=None, splits=None, **kwargs):
    args=['--source',str(source or f['folder']), '--manifest',str(f['manifest']),
          '--mapping',str(f['mapping']), '--output-dir',str(output),
          '--modalities',modality,'--splits',*(splits or ['train','validation'])]
    for key,value in kwargs.items():
        name='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:args.append(name)
        else:args.extend([name,str(value)])
    return run(parse_args(args))


def merge_args(f, inputs, output, splits=None, **kwargs):
    args=['--inputs',*[str(x) for x in inputs], '--manifest',str(f['manifest']),
          '--mapping',str(f['mapping']), '--output-dir',str(output),
          '--splits',*(splits or ['train','validation'])]
    for key,value in kwargs.items():
        name='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:args.append(name)
        else:args.extend([name,str(value)])
    return merge_parser().parse_args(args)


def audit_args(f,output,splits=None,**kwargs):
    args=['--root',str(output),'--manifest',str(f['manifest']),
          '--mapping',str(f['mapping']), '--splits',*(splits or ['train','validation'])]
    for key,value in kwargs.items():
        name='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:args.append(name)
        else:args.extend([name,str(value)])
    return audit_parser().parse_args(args)


def test_three_laptops_merge_and_dataloader(tmp_path):
    f=fixture(tmp_path)
    laptops=[]
    for modality in ['sbsmi','entropy','raw_byte']:
        root=tmp_path/'laptops'/modality
        gen(f,modality,root)
        laptops.append(root)
    result=merge(merge_args(f,laptops,tmp_path/'merged'))
    assert result['selected']==3 and result['images']==9
    assert audit(audit_args(f,tmp_path/'merged',check_extras=True))['pass']
    for split,month,sha in [('train','2019-10','a'*64),('train','2019-11','d'*64),
                            ('validation','2020-03','b'*64)]:
        for modality in ('raw_byte','entropy','sbsmi'):
            assert output_paths(tmp_path/'merged', {'split':split,'month':month,'sha':sha}, modality)[0].is_file()
    assert not (tmp_path/'merged'/'test').exists()
    dataset=TemporalMultiViewDataset(tmp_path/'merged',f['manifest'],f['mapping'],split='train')
    assert len(dataset)==2 and dataset.num_classes==51
    views,y=dataset[0]
    assert all(v.shape==(1,64,64) and v.dtype==__import__('torch').float32 for v in views.values())
    assert all(0<=float(v.min()) <= float(v.max()) <=1 for v in views.values())
    assert y in (0,1)
    with pytest.raises(ValueError,match='explicitly enabled'):
        TemporalMultiViewDataset(tmp_path/'merged',f['manifest'],f['mapping'],split='test')


def test_zip_folder_identical_and_cache_regenerates(tmp_path):
    f=fixture(tmp_path)
    x=tmp_path/'out_folder'; z=tmp_path/'out_zip'
    a=gen(f,'entropy',x)
    b=gen(f,'entropy',z,source=f['archive'])
    assert a['config_id']==b['config_id']
    for sha,month,split in [('a'*64,'2019-10','train'),('b'*64,'2020-03','validation'),('d'*64,'2019-11','train')]:
        row={'split':split,'month':month,'sha':sha}
        p,_=output_paths(x,row,'entropy');q,_=output_paths(z,row,'entropy')
        assert p.read_bytes()==q.read_bytes()
    assert gen(f,'entropy',x)['counts']['VERIFIED']==3
    damaged,_=output_paths(x,{'split':'train','month':'2019-10','sha':'a'*64},'entropy')
    damaged.write_bytes(b'corruption')
    r=gen(f,'entropy',x)
    assert r['counts']=={'CREATED':1,'VERIFIED':2,'ERROR':0}


def test_merge_rejects_missing_modality(tmp_path):
    f=fixture(tmp_path)
    a=tmp_path/'raw';b=tmp_path/'entropy'
    gen(f,'raw_byte',a);gen(f,'entropy',b)
    with pytest.raises(ValueError, match='Expected ONE sbsmi'):
        merge(merge_args(f,[a,b],tmp_path/'merged'))
    assert not (tmp_path/'merged').exists()


def test_merge_rejects_different_source_bytes(tmp_path):
    f=fixture(tmp_path)
    roots=[]
    for m in ('raw_byte','entropy','sbsmi'):
        out=tmp_path/m
        if m=='entropy':
            altered=tmp_path/'changed';altered.mkdir()
            for p in f['folder'].glob('*'):
                altered.joinpath(p.name).write_bytes(p.read_bytes())
            target=altered/('a'*64+'.exe')
            target.write_bytes(bytes([0xff])*target.stat().st_size)
            gen(f,m,out,source=altered)
        else:
            gen(f,m,out)
        roots.append(out)
    with pytest.raises(ValueError,match='DIFFERENT bytes'):
        merge(merge_args(f,roots,tmp_path/'merged'))


def test_future_test_guard_and_merge(tmp_path):
    f=fixture(tmp_path)
    roots=[];config_id=None
    for m in ('raw_byte','entropy','sbsmi'):
        out=tmp_path/m
        result=gen(f,m,out)
        config_id=result['config_id']
        with pytest.raises(ValueError,match='frozen-config-id'):
            gen(f,m,out,splits=['test'])
        gen(f,m,out,splits=['test'],frozen_config_id=config_id)
        roots.append(out)
    merged=tmp_path/'merged'
    merge(merge_args(f,roots,merged))
    with pytest.raises(ValueError,match='frozen-config-id'):
        merge(merge_args(f,roots,merged,splits=['test']))
    merge(merge_args(f,roots,merged,splits=['test'],frozen_config_id=config_id))
    assert audit(audit_args(f,merged,splits=['test']))['pass']
    assert len(TemporalMultiViewDataset(merged,f['manifest'],f['mapping'],split='test',allow_test=True))==1


def test_sharding_assigns_disjoint_samples(tmp_path):
    f=fixture(tmp_path)
    roots=[]
    for shard in (0,1):
        root=tmp_path/f'shard_{shard}'
        # This test chooses b/d vs a parity; both shards non-empty.
        gen(f,'raw_byte',root,shard_id=shard,num_shards=2)
        roots.append(root)
    # Also create separate modality roots; merger finds one source per SHA/view.
    ent=tmp_path/'entropy';sbs=tmp_path/'sbsmi'
    gen(f,'entropy',ent); gen(f,'sbsmi',sbs)
    assert merge(merge_args(f,[*roots,ent,sbs],tmp_path/'merged'))['selected']==3


def test_invalid_split_date_is_rejected(tmp_path):
    f=fixture(tmp_path)
    rows=list(csv.DictReader(f['manifest'].open()))
    rows[0]['month']='2020-09'
    csv_write(f['manifest'],rows)
    with pytest.raises(ValueError,match='Month mismatch'):
        gen(f,'entropy',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_manifest_builder_from_raw_metadata(tmp_path,monkeypatch):
    f=fixture(tmp_path)
    metadata=tmp_path/'metadata.csv'
    entries=[{'sha':sha,'family':family,'timestamp':ts} for sha,split,ts,family,data in f['records']]
    entries.append({'sha':'f'*64,'family':'unknown_family','timestamp':'2020-07-01'})
    csv_write(metadata,entries)
    out=tmp_path/'canonical.csv'
    monkeypatch.setattr('sys.argv',['builder','--metadata',str(metadata),'--mapping',str(f['mapping']),
                                    '--output',str(out)])
    build_main()
    rows=load_manifest(out,load_mapping(f['mapping']))
    assert len(rows)==4
    assert {r['split'] for r in rows}=={'train','validation','test'}


def test_stream_algorithms_chunk_boundary_independence():
    import io
    from src.preprocessing.entropy_image_stream import stream_to_entropy_image
    from src.preprocessing.raw_byte_stream import stream_to_raw_byte
    from src.preprocessing.implementation_sbsmi import stream_to_sbsmi
    rng=np.random.default_rng(321)
    for size in (1,2,127,256,257,512,12800,16385):
        payload=rng.integers(0,256,size,dtype=np.uint8).tobytes()
        baseline=(stream_to_raw_byte(io.BytesIO(payload),size,65536),
                  stream_to_entropy_image(io.BytesIO(payload),size,65536),
                  stream_to_sbsmi(io.BytesIO(payload),bit_num=6,chunk_size=65536))
        for chunk in (1,7,128,257,4096):
            new=(stream_to_raw_byte(io.BytesIO(payload),size,chunk),
                 stream_to_entropy_image(io.BytesIO(payload),size,chunk),
                 stream_to_sbsmi(io.BytesIO(payload),bit_num=6,chunk_size=chunk))
            for expected,actual in zip(baseline,new):
                assert np.array_equal(expected,actual)


def test_missing_binary_preflight_no_output(tmp_path):
    f=fixture(tmp_path)
    (f['folder']/('a'*64+'.exe')).unlink()
    with pytest.raises(ValueError,match='Missing/ambiguous binary'):
        gen(f,'raw_byte',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_conflicting_laptop_code_configs_rejected(tmp_path):
    f=fixture(tmp_path)
    raw=tmp_path/'raw';ent=tmp_path/'ent';sbs=tmp_path/'sbs'
    for m,p in [('raw_byte',raw),('entropy',ent),('sbsmi',sbs)]:
        gen(f,m,p)
    config_path=ent/'config.json'
    config=json.loads(config_path.read_text())
    config['fixed_preprocessing']['entropy']='different algorithm'
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='different preprocessing configurations'):
        merge(merge_args(f,[raw,ent,sbs],tmp_path/'merged'))


def test_duplicate_modality_sources_rejected(tmp_path):
    f=fixture(tmp_path)
    raw1=tmp_path/'raw1';raw2=tmp_path/'raw2';ent=tmp_path/'ent';sbs=tmp_path/'sbs'
    for m,p in [('raw_byte',raw1),('raw_byte',raw2),('entropy',ent),('sbsmi',sbs)]:
        gen(f,m,p)
    with pytest.raises(ValueError,match='Expected ONE raw_byte'):
        merge(merge_args(f,[raw1,raw2,ent,sbs],tmp_path/'merged'))
