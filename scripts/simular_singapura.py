"""Previsão congelada antes do evento e Monte Carlo com as regras públicas do PGL Singapore."""
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reconstruir_major import PROJECT, base_v1, canonical_name, nickname, save_json, hash_file


def snapshots(configuration, source_folder):
    _, series = base_v1()
    cutoff = pd.Timestamp(configuration['data_previsao'])
    series = series.loc[series['datetime'].lt(cutoff)].copy()
    maps = pd.read_csv(PROJECT / 'dados_externos/reconstrucao/df_mapas.csv', parse_dates=['datetime'])
    maps = maps.loc[maps['datetime'].lt(cutoff)]
    vocabulary = json.loads((PROJECT / 'dados_externos/reconstrucao/auditoria_mapas.json').read_bytes())['vocabulario_aprendido_no_treino']
    ranks = json.loads((source_folder / 'ranking_normalizado.json').read_bytes())
    rank_date = pd.Timestamp(configuration['ranking_publicado'])
    if not rank_date < cutoff or (cutoff-rank_date).days > 14:
        raise ValueError('Ranking indisponível ou antigo demais na data da previsão.')
    local_names = set(series['team1']) | set(series['team2'])
    aliases = {'pvision':'parivision', 'gosh':'bcgame', '1win':'1win', 'betboom_team':'betboom', '9z_globant':'9z'}
    values, audit = {}, []
    for participant in configuration['participantes']:
        key = participant['chave']
        roster = {player.casefold() for player in participant['jogadores']}
        desired = {canonical_name(key.replace('_',' ')), canonical_name(participant['nome_oficial'])}
        if key in aliases:
            desired.add(aliases[key])
        compatible_ranks = []
        for rank in ranks:
            common = roster & {player.casefold() for player in rank['jogadores']}
            if canonical_name(rank['nome_hltv']) in desired and len(common) >= 3:
                compatible_ranks.append((rank, common))
        rank = compatible_ranks[0][0] if len(compatible_ranks)==1 else None
        if rank:
            desired.add(canonical_name(rank['nome_hltv']))
        candidates = []
        for local_name in local_names:
            if canonical_name(local_name) not in desired:
                continue
            games = series.loc[(series['team1'].eq(local_name) | series['team2'].eq(local_name)) & series['datetime'].ge(cutoff-pd.Timedelta(days=60))].sort_values('datetime',ascending=False)
            for game in games.head(5).to_dict('records'):
                side = 1 if game['team1']==local_name else 2
                local_roster = {nick for i in range(1,6) if (nick:=nickname(game.get(f'team{side}_player{i}')))}
                common = roster & local_roster
                if len(common)>=3:
                    candidates.append((game['datetime'],local_name,len(common),game['match_id']))
                    break
        candidates.sort(reverse=True)
        # Nunca juntar dois nomes locais: a V1 acumula pelo nome exato.
        local = candidates[0][1] if candidates else None
        history = series.loc[series['team1'].eq(local) | series['team2'].eq(local)] if local else series.iloc[:0]
        wins = int(((history['team1'].eq(local) & history['team1_win_corrigido'].eq(1)) | (history['team2'].eq(local) & history['team1_win_corrigido'].eq(0))).sum())
        record = {'jogos_anteriores':len(history) if local else np.nan,'vitorias_anteriores':wins if local else np.nan,'taxa_vitorias_anteriores':wins/len(history) if len(history) else np.nan,'pontos_hltv':rank['pontos'] if rank else np.nan}
        for map_name in vocabulary:
            played = maps.loc[maps['map_name'].eq(map_name) & (maps['team1'].eq(local) | maps['team2'].eq(local))] if local else maps.iloc[:0]
            map_wins = int(((played['team1'].eq(local) & played['team1_win_mapa'].eq(1)) | (played['team2'].eq(local) & played['team1_win_mapa'].eq(0))).sum())
            record['jogos_anteriores_'+map_name.lower()] = len(played) if local else np.nan
            record['taxa_vitorias_anteriores_'+map_name.lower()] = map_wins/len(played) if len(played) else np.nan
        values[key] = record
        audit.append({**{column:participant[column] for column in ['chave','nome_oficial','etapa_entrada','seed_entrada']},'nome_local':local,'hltv_team_id':rank['hltv_team_id'] if rank else None,'nome_hltv':rank['nome_hltv'] if rank else None,'ranking_data':configuration['ranking_publicado'],'ranking_pontos':rank['pontos'] if rank else None,'jogadores_em_comum_ranking':len(compatible_ranks[0][1]) if rank else 0,'jogadores_em_comum_local':candidates[0][2] if candidates else 0,'confronto_evidencia_local':candidates[0][3] if candidates else None,'data_evidencia_local':candidates[0][0] if candidates else None,'jogos_historico_local':len(history),'ultima_partida_local':str(history['datetime'].max()) if len(history) else None,'status_local':'corroborado_por_nome_e_escalacao' if local else 'sem_identidade_local_corroborada','status_ranking':'corroborado_por_nome_e_escalacao' if rank else 'sem_correspondencia_corroborada'})
    return values, pd.DataFrame(audit), vocabulary


def probability_matrices(configuration, values, model, columns):
    keys=[participant['chave'] for participant in configuration['participantes']]
    matrices, rows = {}, []
    for bestof in [1,3,5]:
        matrix=np.full((32,32),0.5)
        pairs, inputs=[],[]
        for i,j in itertools.permutations(range(32),2):
            first,second=values[keys[i]],values[keys[j]]
            inputs.append({column:bestof if column=='bestOf' else first[column.removeprefix('dif_')]-second[column.removeprefix('dif_')] for column in columns})
            pairs.append((i,j))
        x=pd.DataFrame(inputs,columns=columns)
        probabilities=model.predict_proba(x)[:,list(model.classes_).index(1)]
        for (i,j),probability in zip(pairs,probabilities):
            matrix[i,j]=probability
            if i<j:
                rows.append({'equipe_a':keys[i],'equipe_b':keys[j],'bestOf':bestof,'prob_vitoria_a':float(probability),'prob_vitoria_b':float(1-probability)})
        if not np.isfinite(matrix).all() or not ((matrix>=0)&(matrix<=1)).all() or not np.allclose(matrix+matrix.T,1,atol=1e-12):
            raise ValueError('As probabilidades não são válidas e complementares ao inverter as equipes.')
        matrices[bestof]=matrix
    return keys,matrices,pd.DataFrame(rows)


def pair_without_rematches(ordered, opponents, priority):
    if len(ordered)==6:
        for pattern in priority:
            pairs=[(ordered[a-1],ordered[b-1]) for a,b in pattern]
            if all(b not in opponents[a] for a,b in pairs):
                return pairs
        raise ValueError('Não foi possível aplicar a tabela de prioridades de seis equipes.')
    def search(remaining):
        if not remaining:
            return []
        first=remaining[0]
        for position in range(len(remaining)-1,0,-1):
            opponent=remaining[position]
            if opponent in opponents[first]:
                continue
            rest=search(remaining[1:position]+remaining[position+1:])
            if rest is not None:
                return [(first,opponent)]+rest
        return None
    result=search(ordered)
    if result is None:
        raise ValueError('Pareamento exigiria revanche dentro da etapa.')
    return result


def swiss_stage(entrants, stage, matrices, rng, priority, register):
    if len(entrants)!=16 or len(set(entrants))!=16:
        raise ValueError('Etapa suíça precisa de 16 equipes distintas.')
    seeds={team:index for index,team in enumerate(entrants)}
    records={team:[0,0] for team in entrants}
    opponents={team:[] for team in entrants}
    def buchholz(team):
        return sum(records[other][0]-records[other][1] for other in opponents[team])
    def ordered(group):
        return sorted(group,key=lambda team:(-buchholz(team),seeds[team]))
    total_matches=0
    for round_number in range(5):
        pairs=[]
        if round_number==0:
            pairs=[(entrants[index],entrants[index+8]) for index in range(8)]
        else:
            groups={}
            for team,(wins,losses) in records.items():
                if wins<3 and losses<3:
                    groups.setdefault((wins,losses),[]).append(team)
            # Definir todos os jogos antes de atualizar os resultados da rodada.
            for record,group in sorted(groups.items(),reverse=True):
                pairs.extend(pair_without_rematches(ordered(group),opponents,priority))
        outcomes=[]
        for first,second in pairs:
            if records[first]!=records[second] or second in opponents[first]:
                raise ValueError('Campanhas incompatíveis ou revanche no suíço.')
            wins,losses=records[first]
            bestof=3 if stage==3 or wins==2 or losses==2 else 1
            winner=first if rng.random()<matrices[bestof][first,second] else second
            loser=second if winner==first else first
            outcomes.append((first,second,winner,loser))
            register(first,second,stage)
        for first,second,winner,loser in outcomes:
            records[winner][0]+=1
            records[loser][1]+=1
            opponents[first].append(second)
            opponents[second].append(first)
        total_matches+=len(pairs)
    qualifiers=[team for team,record in records.items() if record[0]==3]
    if len(qualifiers)!=8 or total_matches!=33 or any(sum(record)>5 or (record[0]!=3 and record[1]!=3) for record in records.values()):
        raise ValueError('Resultado inválido da etapa suíça.')
    return sorted(qualifiers,key=lambda team:(records[team][1],-buchholz(team),seeds[team]))


def monte_carlo(configuration,keys,matrices,iterations=20000,seed=25):
    rng=np.random.default_rng(seed)
    positions={key:index for index,key in enumerate(keys)}
    stages={stage:[positions[key] for key in configuration['etapas'][str(stage)]] for stage in [1,2,3]}
    counts=np.zeros((32,7),dtype=np.int64)
    meetings=np.zeros((32,32),dtype=np.int64)
    by_stage={stage:np.zeros((32,32),dtype=np.int64) for stage in [1,2,3,4]}
    for iteration in range(iterations):
        seen=set()
        def register(first,second,stage):
            seen.add(tuple(sorted((first,second))))
            by_stage[stage][first,second]+=1
            by_stage[stage][second,first]+=1
        q1=swiss_stage(stages[1],1,matrices,rng,configuration['tabela_prioridade_grupos_seis'],register)
        entrants2=stages[2]+q1
        q2=swiss_stage(entrants2,2,matrices,rng,configuration['tabela_prioridade_grupos_seis'],register)
        entrants3=stages[3]+q2
        q3=swiss_stage(entrants3,3,matrices,rng,configuration['tabela_prioridade_grupos_seis'],register)
        counts[stages[2]+stages[3]+q1,0]+=1
        counts[entrants3,1]+=1
        counts[q3,2]+=1
        def play(first,second,bestof):
            register(first,second,4)
            return (first,second) if rng.random()<matrices[bestof][first,second] else (second,first)
        quarter_winners=[play(q3[a-1],q3[b-1],3)[0] for a,b in configuration['regras']['quartas']]
        counts[quarter_winners,3]+=1
        first_finalist,first_loser=play(quarter_winners[0],quarter_winners[1],3)
        second_finalist,second_loser=play(quarter_winners[2],quarter_winners[3],3)
        counts[[first_finalist,second_finalist],4]+=1
        third_place,_=play(first_loser,second_loser,3)
        counts[third_place,6]+=1
        champion,_=play(first_finalist,second_finalist,5)
        counts[champion,5]+=1
        for first,second in seen:
            meetings[first,second]+=1
            meetings[second,first]+=1
        if (iteration+1)%5000==0:
            print(f'Simulacoes: {iteration+1}/{iterations}',flush=True)
    if not np.array_equal(counts.sum(axis=0),np.array([24,16,8,4,2,1,1])*iterations):
        raise ValueError('As probabilidades não conservam o número de vagas e títulos.')
    return counts,meetings,by_stage


def main():
    pointer=json.loads((PROJECT/'singapura_2026/ultima_preparacao.json').read_bytes())
    source_folder=Path(pointer['pasta_fontes'])
    configuration=json.loads((source_folder/'config_torneio.json').read_bytes())
    report=json.loads((PROJECT/'dados_externos/reconstrucao/ultimo_experimento.json').read_bytes())
    experiment=Path(report['pasta_experimento'])
    selection=json.loads((experiment/'selecao_antes_do_teste.json').read_bytes())
    model=joblib.load(experiment/'modelo_selecionado.joblib')
    model_hash=hash_file(PROJECT/'modelos/modelo_v1.joblib')
    output=PROJECT/'singapura_2026'/('simulacao_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True,exist_ok=False)
    values,audit,vocabulary=snapshots(configuration,source_folder)
    audit.to_csv(output/'participantes_e_cobertura.csv',index=False)
    keys,matrices,_=probability_matrices(configuration,values,model,selection['entradas'])
    # Verificar o motor com um caso determinístico: a equipe de índice zero vence todas.
    deterministic={bo:(np.arange(32)[:,None]<np.arange(32)[None,:]).astype(float) for bo in [1,3,5]}
    check,_,_=monte_carlo(configuration,keys,deterministic,iterations=3,seed=25)
    if check[0,5]!=3:
        raise ValueError('O motor falhou no caso de campeão determinístico.')
    iterations=20000
    counts,meetings,by_stage=monte_carlo(configuration,keys,matrices,iterations=iterations,seed=25)
    columns=['alcancar_etapa2_ou_superior','alcancar_etapa3_ou_superior','chegar_playoffs','chegar_semifinal','chegar_final','ser_campea','terceiro_lugar']
    chances=pd.DataFrame(counts/iterations,index=keys,columns=columns).reset_index(names='chave').merge(audit[['chave','nome_oficial','etapa_entrada']],on='chave',validate='one_to_one')
    # Intervalo de Wilson: mede só o erro de Monte Carlo, não a incerteza do modelo.
    p=chances['ser_campea']; z=1.96
    center=(p+z*z/(2*iterations))/(1+z*z/iterations)
    margin=z*np.sqrt(p*(1-p)/iterations+z*z/(4*iterations*iterations))/(1+z*z/iterations)
    chances['campea_ic95_mc_inferior']=center-margin
    chances['campea_ic95_mc_superior']=center+margin
    chances=chances.sort_values('ser_campea',ascending=False)
    chances.to_csv(output/'chances_por_equipe.csv',index=False)
    initial=[]
    for index in range(8):
        first,second=configuration['etapas']['1'][index],configuration['etapas']['1'][index+8]
        p=float(matrices[1][keys.index(first),keys.index(second)])
        initial.append({'equipe_a':first,'equipe_b':second,'bestOf':1,'prob_vitoria_a':p,'prob_vitoria_b':1-p})
    pd.DataFrame(initial).to_csv(output/'primeira_rodada.csv',index=False)
    if hash_file(PROJECT/'modelos/modelo_v1.joblib')!=model_hash:
        raise ValueError('O modelo V1 mudou durante a simulação.')
    summary={'evento':configuration['evento'],'data_previsao':configuration['data_previsao'],'dados_locais_ate':'2026-11-14','ranking_publicado':configuration['ranking_publicado'],'modelo':report['melhor_log_loss_validacao'],'pasta_modelo':str(experiment),'pasta_simulacao':str(output),'pasta_fontes':str(source_folder),'simulacoes':iterations,'semente_aleatoria':25,'equipes_com_identidade_local_corroborada':int(audit['nome_local'].notna().sum()),'equipes_com_ranking_corroborado':int(audit['hltv_team_id'].notna().sum()),'vocabulario_mapas_treino':vocabulary,'invariantes_conferidos':['32 participantes distintos','campanhas iguais e nenhuma revanche dentro de cada suíço','33 séries e 8 classificados por suíço','etapas 1/2: avanço e eliminação em MD3; etapa 3 toda MD3','8/4/2/1 equipes nos playoffs, semifinal, final e título','probabilidades complementares ao inverter as equipes','campeão determinístico no caso de controle'],'hipoteses':['Históricos congelados antes do evento; resultados simulados não alteram as entradas.','Séries modeladas diretamente por bestOf, sem usar mapas do veto futuro.','Identidade requer nome compatível e três jogadores corroborantes; ausências permanecem explícitas.','Probabilidades refletem o modelo, a amostra e as regras arquivadas; intervalo de Monte Carlo não mede erro do modelo.','Dados locais param em 14/11; partidas posteriores a essa data não foram acrescentadas.'],'top_5':chances[['nome_oficial','ser_campea']].head(5).to_dict('records'),'primeira_rodada':initial,'modelo_v1_sha256':model_hash,'selecao_modelo_sha256':hash_file(experiment/'selecao_antes_do_teste.json')}
    save_json(PROJECT/'singapura_2026/ultima_simulacao.json',summary)
    print(json.dumps(summary,ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
