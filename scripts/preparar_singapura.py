"""Prepara somente a configuração e o ranking usados pela simulação do Major."""
import ast
import contextlib
import hashlib
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from scrapy.http import HtmlResponse

from executar_piloto_hltv import normalize, EXPECTED_COMMIT

PROJECT = Path(r'C:\Users\roger.peloso\Desktop\major')
COLLECTOR = PROJECT.parent / 'hltv-scraper-api'
sys.path.insert(0, str(COLLECTOR))
from hltv_scraper.hltv_scraper.spiders.hltv_top30 import HltvTop30Spider


def literal_at(text, start):
    """Lê somente literais de dados; nenhum JavaScript remoto é executado."""
    if text[start] not in '{[':
        raise ValueError('Estrutura da página mudou; revisar o extrator.')
    depth, quote, escaped = 0, None, False
    for end in range(start, len(text)):
        char = text[end]
        if quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
        elif char in '\"\'':
            quote = char
        elif char in '{[':
            depth += 1
        elif char in '}]':
            depth -= 1
            if depth == 0:
                raw = text[start:end+1]
                quoted = re.sub(r'([{,])\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*:', r'\1"\2":', raw)
                return ast.literal_eval(quoted)
    raise ValueError('Literal incompleto na página.')


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    now = datetime.now(timezone.utc)
    output = PROJECT / 'singapura_2026' / ('fontes_' + now.strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True, exist_ok=False)
    url = 'https://www.pglesports.com/cs2/pgl-major-singapore-2026/'
    session = requests.Session()
    page = session.get(url, timeout=30)
    page.raise_for_status()
    script_path = re.search(r'<script[^>]+src="([^"]*app/cs2/pgl-major-singapore-2026/page-[^"]+\.js)"', page.text)
    if not script_path:
        raise ValueError('Arquivo de dados do evento não encontrado na página oficial.')
    script_url = urljoin(url, script_path.group(1))
    response = session.get(script_url, timeout=30)
    response.raise_for_status()
    text = response.text
    stages = {str(key): value for key, value in literal_at(text, text.index('D={') + 2).items()}
    rosters = literal_at(text, text.index('var Ne={') + 7)
    index = text.index('ninjas_in_pyjamas:["NIP"')
    names = literal_at(text, text.rfind('Object.entries(', 0, index) + len('Object.entries('))
    priority = literal_at(text, text.index('var Ee=[') + 7)
    participants = []
    for stage, teams in stages.items():
        for seed, team in enumerate(teams, 1):
            participants.append({'chave':team, 'nome_oficial':names[team][1], 'etapa_entrada':int(stage), 'seed_entrada':seed, 'jogadores':rosters[team]['players'], 'reserva':rosters[team].get('sub')})
    if len(participants) != 32 or len({row['chave'] for row in participants}) != 32:
        raise ValueError('Lista oficial não contém 32 participantes distintos.')
    if [len(stages[str(stage)]) for stage in [1,2,3]] != [16,8,8]:
        raise ValueError('Distribuição por etapas mudou.')
    ranking_url = 'https://www.hltv.org/ranking/teams/2026/november/16'
    ranking_response = session.get(ranking_url, timeout=40, headers={'User-Agent':'hltv_scraper (+https://www.hltv.org/)'})
    ranking_response.raise_for_status()
    with contextlib.redirect_stdout(io.StringIO()):
        ranking = next(HltvTop30Spider(year='2026', month='november', day=16).parse(HtmlResponse(url=ranking_url,body=ranking_response.content,encoding='utf-8')))
    normalized = normalize(ranking,publication='2026-11-16',source=ranking_url)
    players = {int(team['id']):team['players'] for team in ranking['ranking']}
    for row in normalized:
        row['jogadores'] = players[row['hltv_team_id']]
    save(output / 'ranking_normalizado.json',normalized)
    configuration = {'evento':'PGL Major Singapore 2026','fonte_oficial':url,'arquivo_dados_oficial':script_url,'consultado_em_utc':now.isoformat(),'data_previsao':'2026-11-17','inicio_evento':'2026-11-24','fim_evento':'2026-12-13','ranking_publicado':'2026-11-16','participantes':participants,'etapas':stages,'tabela_prioridade_grupos_seis':priority,'regras':{'avanca_com_vitorias':3,'eliminado_com_derrotas':3,'rodadas_maximas_suico':5,'etapas_1_2':'BO1; avanço ou eliminação em BO3','etapa_3':'todas as séries em BO3','desempates':['campanha','Buchholz','seed inicial'],'evitar_revanche_na_mesma_etapa':True,'seeds_nova_etapa':'oito convidados seguidos dos oito classificados, na ordem final da etapa anterior','quartas':[[1,8],[4,5],[2,7],[3,6]],'eliminatorias':'quartas, semifinais e terceiro lugar BO3; final BO5'},'commit_parser_hltv':EXPECTED_COMMIT,'pagina_sha256':hashlib.sha256(page.content).hexdigest(),'dados_oficiais_sha256':hashlib.sha256(response.content).hexdigest()}
    save(output / 'config_torneio.json',configuration)
    save(PROJECT / 'singapura_2026' / 'ultima_preparacao.json',{'pasta_fontes':str(output)})
    print(json.dumps({'pasta_fontes':str(output),'participantes':32,'ranking_equipes':len(normalized),'etapas':{key:len(value) for key,value in stages.items()}},ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
