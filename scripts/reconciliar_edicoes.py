"""Completa edições referenciadas pelos links oficiais, inclusive publicações fora da segunda-feira."""
import contextlib
import hashlib
import io
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from scrapy.http import HtmlResponse

from coletar_rankings_historicos import ROOT, COLLECTOR, MONTHS, EXPECTED_COMMIT, save_json
from executar_piloto_hltv import normalize

sys.path.insert(0, str(COLLECTOR))
from hltv_scraper.hltv_scraper.spiders.hltv_top30 import HltvTop30Spider


def parse_link(link):
    year, month, day = link.split("/")
    return date(int(year), MONTHS.index(month.lower()) + 1, int(day))


def main():
    start, end = date(2023, 1, 9), date(2026, 11, 14)
    session = requests.Session()
    session.headers['User-Agent'] = 'hltv_scraper (+https://www.hltv.org/)'
    attempted = set()
    for _ in range(10):
        existing, referenced = {}, set()
        for path in ROOT.glob('????-??-??/ranking_bruto.json'):
            raw = json.loads(path.read_bytes())
            metadata = json.loads((path.parent / 'metadados.json').read_bytes())
            if metadata.get('status') != 'conteudo_validado':
                continue
            existing[raw['date']] = (path, raw, metadata)
            for field in ['prev_ranking', 'next_ranking']:
                if raw.get(field):
                    linked_date = parse_link(raw[field])
                    if start <= linked_date <= end:
                        referenced.add(linked_date.isoformat())
        missing = referenced - existing.keys() - attempted
        if not missing:
            break
        for iso_date in sorted(missing):
            attempted.add(iso_date)
            publication = date.fromisoformat(iso_date)
            folder = ROOT / iso_date
            folder.mkdir(exist_ok=True)
            url = f'https://www.hltv.org/ranking/teams/{publication.year}/{MONTHS[publication.month-1]}/{publication.day}'
            metadata = {'data_solicitada':iso_date, 'url_fonte':url, 'coletado_em_utc':datetime.now(timezone.utc).isoformat(), 'commit_coletor':EXPECTED_COMMIT, 'transporte':'requests sequencial; parser original HltvTop30Spider; descoberta pelos links entre publicacoes'}
            try:
                response=session.get(url,timeout=(10,40))
                metadata['http_status']=response.status_code
                metadata['html_sha256']=hashlib.sha256(response.content).hexdigest()
                response.raise_for_status()
                with contextlib.redirect_stdout(io.StringIO()):
                    raw=next(HltvTop30Spider(year=str(publication.year),month=MONTHS[publication.month-1],day=publication.day).parse(HtmlResponse(url=response.url,body=response.content,encoding='utf-8')))
                save_json(folder/'ranking_bruto.json',raw)
                normalized=normalize(raw,publication=iso_date,source=url)
                players={int(team['id']):team['players'] for team in raw['ranking']}
                for row in normalized:
                    row.update(jogadores=players[row['hltv_team_id']],coletado_em_utc=metadata['coletado_em_utc'],commit_coletor=EXPECTED_COMMIT)
                metadata.update(status='conteudo_validado',data_recebida=raw['date'],equipes=len(normalized))
                print('Edicao adicional validada:',iso_date,len(normalized),flush=True)
            except Exception as error:
                metadata.update(status='falha',erro=f'{type(error).__name__}: {error}')
                print('FALHA',iso_date,metadata['erro'],flush=True)
            if metadata.get('status')=='conteudo_validado':
                save_json(folder/'metadados.json',metadata)
            time.sleep(3)
    records,index,failures=[],[],[]
    for path in sorted(ROOT.glob('????-??-??/metadados.json')):
        metadata=json.loads(path.read_bytes())
        if metadata.get('status')=='conteudo_validado':
            raw=json.loads((path.parent/'ranking_bruto.json').read_bytes())
            normalized=normalize(raw,publication=metadata['data_recebida'],source=metadata['url_fonte'])
            players={int(team['id']):team['players'] for team in raw['ranking']}
            for row in normalized:
                row.update(jogadores=players[row['hltv_team_id']],coletado_em_utc=metadata['coletado_em_utc'],commit_coletor=EXPECTED_COMMIT)
            records.extend(normalized)
            index.append(metadata)
        else:
            failures.append(metadata)
    existing={item['data_recebida'] for item in index}
    unresolved=sorted(referenced-existing)
    summary={'edicoes_planejadas_no_calendario':201,'edicoes_validadas':len(index),'consultas_sem_edicao_ou_com_falha':len(failures),'edicoes_referenciadas_ainda_ausentes':unresolved,'linhas':len(records),'primeira_edicao':min(existing),'ultima_edicao':max(existing)}
    save_json(ROOT/'rankings_historicos.json',records)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
