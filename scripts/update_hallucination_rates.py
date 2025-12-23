import json
import os
import io
import csv
from pathlib import Path

def update_hallucination_rates():
    # Paths
    project_root = Path("/Users/annette/repostitories/llm_jury")
    models_path = project_root / "final_release/models.json"
    cache_path = project_root / "banditgpt/data/models_cache.json"
    
    # Load models
    with open(models_path, "r") as f:
        models_data = json.load(f)
    
    with open(cache_path, "r") as f:
        cache_data = json.load(f)
    
    # User provided CSV data
    csv_raw = """modelName,omniscienceHallucinationRate,detailsUrl,isLabClaimedValue
Jamba 1.7 Large,1,/models/jamba-1-7-large/providers,false
Gemma 3 4B,0.9807830459770115,/models/gemma-3-4b/providers,false
Jamba 1.7 Mini,0.9714285714285714,/models/jamba-1-7-mini/providers,false
Qwen3 Omni 30B A3B,0.9700193423597679,/models/qwen3-omni-30b-a3b-instruct/providers,false
Gemma 3 12B,0.9683509161576902,/models/gemma-3-12b/providers,false
Qwen3 1.7B,0.9679856115107913,/models/qwen3-1.7b-instruct/providers,false
Gemma 3n E4B,0.9656165616561656,/models/gemma-3n-e4b/providers,false
Granite 4.0 H 350M,0.962707182320442,/models/granite-4-0-h-350m/providers,false
Qwen3 8B,0.955043655953929,/models/qwen3-8b-instruct/providers,false
Granite 4.0 Micro,0.9538068285557787,/models/granite-4-0-micro/providers,false
Qwen3 0.6B,0.9504175365344467,/models/qwen3-0.6b-instruct/providers,false
Qwen3 1.7B,0.946698199017646,/models/qwen3-1.7b-instruct-reasoning/providers,false
Qwen3 30B A3B 2507,0.9463243873978997,/models/qwen3-30b-a3b-2507/providers,false
Granite 4.0 1B,0.9443952541172304,/models/granite-4-0-nano-1b/providers,false
Exaone 4.0 1.2B,0.9432019879304224,/models/exaone-4-0-1-2b-reasoning/providers,false
Ling-flash-2.0,0.9410741885625966,/models/ling-flash-2-0/providers,false
Hermes 4 70B,0.9397254397254398,/models/hermes-4-llama-3-1-70b-reasoning/providers,false
NVIDIA Nemotron Nano 12B v2 VL,0.9391564016424039,/models/nvidia-nemotron-nano-12b-v2-vl/providers,false
Hermes 4 405B,0.9379261697625205,/models/hermes-4-llama-3-1-405b-reasoning/providers,false
OLMo 3 7B Think,0.9365491651205937,/models/olmo-3-7b-think/providers,false
Ling-1T,0.9339622641509434,/models/ling-1t/providers,false
Solar Pro 2,0.9337152209492635,/models/solar-pro-2-reasoning/providers,false
gpt-oss-20B (high),0.9320445225541887,/models/gpt-oss-20b/providers,false
Qwen3 0.6B,0.9317098994176813,/models/qwen3-0.6b-instruct-reasoning/providers,false
GLM-4.6,0.9308879445314248,/models/glm-4-6-reasoning/providers,false
Nova 2.0 Omni (medium),0.928743961352657,/models/nova-2-0-omni-reasoning-medium/providers,false
Ministral 8B (Dec '25),0.9276129276129276,/models/ministral-8b/providers,false
Qwen3 Next 80B A3B,0.9269561737042226,/models/qwen3-next-80b-a3b-instruct/providers,false
Gemma 3n E2B,0.9262353359402773,/models/gemma-3n-e2b/providers,false
Gemini 2.5 Flash,0.9256530475552579,/models/gemini-2-5-flash/providers,false
MiMo-V2-Flash,0.9253393665158371,/models/mimo-v2-flash-reasoning/providers,false
DeepSeek V3.2,0.9251186879585671,/models/deepseek-v3-2/providers,false
Ling-mini-2.0,0.925,/models/ling-mini-2-0/providers,false
Qwen3 VL 30B A3B,0.9241446725317694,/models/qwen3-vl-30b-a3b-instruct/providers,false
GPT-4.1,0.9228861330326945,/models/gpt-4-1/providers,false
GLM-4.5-Air,0.9205414949970571,/models/glm-4-5-air/providers,false
Exaone 4.0 1.2B,0.920321509697711,/models/exaone-4-0-1-2b/providers,false
NVIDIA Nemotron Nano 12B v2 VL,0.9197922677437969,/models/nvidia-nemotron-nano-12b-v2-vl-reasoning/providers,false
Solar Pro 2,0.9197724597881523,/models/solar-pro-2/providers,false
Apriel-v1.6-15B-Thinker,0.916466826538769,/models/apriel-v1-6-15b-thinker/providers,false
GPT-4.1 mini,0.9159146841673503,/models/gpt-4-1-mini/providers,false
OLMo 3 7B,0.915263346470799,/models/olmo-3-7b-instruct/providers,false
DeepSeek V3.2 Exp,0.9133875106928999,/models/deepseek-v3-2-0925/providers,false
Gemini 3 Flash,0.9120879120879121,/models/gemini-3-flash-reasoning/providers,false
Gemma 3 27B,0.9110563246728618,/models/gemma-3-27b/providers,false
Gemini 3 Flash,0.9094922737306843,/models/gemini-3-flash/providers,false
Nova 2.0 Lite (medium),0.9076240419524002,/models/nova-2-0-lite-reasoning-medium/providers,false
Magistral Small 1.2,0.9073366450133741,/models/magistral-small-2509/providers,false
Qwen3 VL 32B,0.9069226294357184,/models/qwen3-vl-32b-instruct/providers,false
gpt-oss-120B (low),0.9051109753614335,/models/gpt-oss-120b-low/providers,false
Ministral 14B (Dec '25),0.904969650986343,/models/ministral-14b/providers,false
Qwen3 Max Thinking,0.9048376107199637,/models/qwen3-max-thinking/providers,false
Qwen3 VL 8B,0.9047521086196256,/models/qwen3-vl-8b-reasoning/providers,false
Qwen3 VL 235B A22B,0.904683309263462,/models/qwen3-vl-235b-a22b-instruct/providers,false
Gemini 2.5 Flash (Sep),0.9036820835204311,/models/gemini-2-5-flash-preview-09-2025/providers,false
Qwen3 8B,0.9035523300229182,/models/qwen3-8b-instruct-reasoning/providers,false
Reka Flash 3,0.903244384718756,/models/reka-flash-3/providers,false
Qwen3 VL 8B,0.902680412371134,/models/qwen3-vl-8b-instruct/providers,false
Nova 2.0 Pro Preview (medium),0.9013282732447818,/models/nova-2-0-pro-reasoning-medium/providers,false
gpt-oss-120B (high),0.899562408835174,/models/gpt-oss-120b/providers,false
NVIDIA Nemotron 3 Nano,0.8981233243967829,/models/nvidia-nemotron-3-nano-30b-a3b/providers,false
GLM-4.5V,0.8968158000806127,/models/glm-4-5v/providers,false
Qwen3 235B A22B 2507,0.8964262786218703,/models/qwen3-235b-a22b-instruct-2507-reasoning/providers,false
K2-V2 (high),0.8949799196787148,/models/k2-v2/providers,false
Qwen3 VL 30B A3B,0.8929421094369548,/models/qwen3-vl-30b-a3b-reasoning/providers,false
GPT-5.1,0.891735918744229,/models/gpt-5-1-non-reasoning/providers,false
DeepSeek V3 (Dec),0.8915715539947322,/models/deepseek-v3/providers,false
MiniMax M1 80k,0.8904741921947126,/models/minimax-m1-80k/providers,false
Qwen3 Max,0.8904109589041096,/models/qwen3-max/providers,false
Motif-2-12.7B,0.8889967009509024,/models/motif-2-12-7b/providers,false
MiniMax-M2,0.8888421052631579,/models/minimax-m2/providers,false
DeepSeek R1 (Jan),0.8879208853308218,/models/deepseek-r1-0120/providers,false
Gemini 2.5 Pro,0.8866968808317782,/models/gemini-2-5-pro/providers,false
Nova 2.0 Omni,0.8858664637626023,/models/nova-2-0-omni/providers,false
DeepSeek V3.2 Speciale,0.8851119894598155,/models/deepseek-v3-2-speciale/providers,false
Gemini 2.5 Flash (Sep),0.883131705090162,/models/gemini-2-5-flash-preview-09-2025-reasoning/providers,false
Qwen3 Omni 30B A3B,0.8824670287044221,/models/qwen3-omni-30b-a3b-reasoning/providers,false
Gemini 3 Pro Preview (high),0.8798993167925206,/models/gemini-3-pro/providers,false
Qwen3 14B,0.8788966559607028,/models/qwen3-14b-instruct/providers,false
GPT-5 (minimal),0.8777192580719029,/models/gpt-5-minimal/providers,false
Qwen3 Max (Preview),0.8770974068424493,/models/qwen3-max-preview/providers,false
GPT-5 nano (minimal),0.8770214366303122,/models/gpt-5-nano-minimal/providers,false
GPT-5 mini (minimal),0.8769601930036188,/models/gpt-5-mini-minimal/providers,false
Llama 4 Maverick,0.8757899324471562,/models/llama-4-maverick/providers,false
Granite 4.0 H Small,0.8725207009435779,/models/granite-4-0-h-small/providers,false
Gemini 2.5 Flash-Lite,0.872211350293542,/models/gemini-2-5-flash-lite/providers,false
Gemini 3 Pro Preview (low),0.8718740351960481,/models/gemini-3-pro-low/providers,false
Devstral Small 2,0.8688492452460302,/models/devstral-small-2/providers,false
DeepSeek V3.1 Terminus,0.8684040491061813,/models/deepseek-v3-1-terminus/providers,false
Qwen3 Next 80B A3B,0.8681475443244345,/models/qwen3-next-80b-a3b-reasoning/providers,false
o3,0.8674634794156707,/models/o3/providers,false
Nova 2.0 Pro Preview (low),0.8666947901286648,/models/nova-2-0-pro-reasoning-low/providers,false
Qwen3 235B,0.8661829907295445,/models/qwen3-235b-a22b-instruct/providers,false
Gemini 2.5 Flash-Lite (Sep),0.8660498793242156,/models/gemini-2-5-flash-lite-preview-09-2025-reasoning/providers,false
Gemma 3 1B,0.8652983787512936,/models/gemma-3-1b/providers,false
Grok 4 Fast,0.864098982239074,/models/grok-4-fast/providers,false
OLMo 3 32B Think,0.8630110443712459,/models/olmo-3-32b-think/providers,false
DeepSeek R1 0528 Qwen3 8B,0.8627230046948356,/models/deepseek-r1-qwen3-8b/providers,false
EXAONE 4.0 32B,0.8625,/models/exaone-4-0-32b-reasoning/providers,false
Qwen3 30B A3B 2507,0.8623817034700315,/models/qwen3-30b-a3b-2507-reasoning/providers,false
Grok 3,0.8623221661312529,/models/grok-3/providers,false
Nova 2.0 Lite (low),0.8612612612612612,/models/nova-2-0-lite-reasoning-low/providers,false
gpt-oss-20B (low),0.8605908476539873,/models/gpt-oss-20b-low/providers,false
Seed-OSS-36B-Instruct,0.8572580645161291,/models/seed-oss-36b-instruct/providers,false
ERNIE 5.0 Thinking Preview,0.8533304404426123,/models/ernie-5-0-thinking-preview/providers,false
DeepSeek V3 0324,0.8518438177874187,/models/deepseek-v3-0324/providers,false
Nova 2.0 Lite,0.8506630789928887,/models/nova-2-0-lite/providers,false
Mistral Large 3,0.8473465822231928,/models/mistral-large-3/providers,false
DeepSeek V3.1,0.8472758472758473,/models/deepseek-v3-1/providers,false
Devstral 2,0.8443474646716542,/models/devstral-2/providers,false
Qwen3 VL 235B A22B,0.8424470982610518,/models/qwen3-vl-235b-a22b-reasoning/providers,false
Nova 2.0 Omni (low),0.8407294832826747,/models/nova-2-0-omni-reasoning-low/providers,false
Apriel-v1.5-15B-Thinker,0.8400236127508854,/models/apriel-v1-5-15b-thinker/providers,false
Granite 4.0 H 1B,0.8359361291454641,/models/granite-4-0-h-nano-1b/providers,false
DeepSeek R1 0528,0.8336082960169692,/models/deepseek-r1/providers,false
GLM-4.5V,0.8332637729549248,/models/glm-4-5v-reasoning/providers,false
Qwen3 VL 32B,0.8322040653646872,/models/qwen3-vl-32b-reasoning/providers,false
Hermes 4 70B,0.8259634888438134,/models/hermes-4-llama-3-1-70b/providers,false
NVIDIA Nemotron 3 Nano,0.8253144340187663,/models/nvidia-nemotron-3-nano-30b-a3b-reasoning/providers,false
Mistral Medium 3.1,0.824799506477483,/models/mistral-medium-3-1/providers,false
EXAONE 4.0 32B,0.8242042931162102,/models/exaone-4-0-32b/providers,false
GPT-4.1 nano,0.8224727689661762,/models/gpt-4-1-nano/providers,false
DeepSeek V3.2,0.8192771084337349,/models/deepseek-v3-2-reasoning/providers,false
GPT-5 (medium),0.8163428267234496,/models/gpt-5-medium/providers,false
DeepSeek V3.1,0.816179879462216,/models/deepseek-v3-1-reasoning/providers,false
Qwen3 32B,0.8143712574850299,/models/qwen3-32b-instruct-reasoning/providers,false
GPT-5 (high),0.8099375509095845,/models/gpt-5/providers,false
Llama Nemotron Ultra,0.8094059405940595,/models/llama-3-1-nemotron-ultra-253b-v1-reasoning/providers,false
DeepSeek R1 Distill Llama 70B,0.808997955010225,/models/deepseek-r1-distill-llama-70b/providers,false
Grok 4.1 Fast,0.8089865399841647,/models/grok-4-1-fast/providers,false
DeepSeek V3.2 Exp,0.8060246462802373,/models/deepseek-v3-2-reasoning-0925/providers,false
K2-V2 (medium),0.8050339592489013,/models/k2-v2-medium/providers,false
Llama 3.2 11B (Vision),0.8027286135693216,/models/llama-3-2-instruct-11b-vision/providers,false
Qwen3 30B,0.7995668438669029,/models/qwen3-30b-a3b-instruct-reasoning/providers,false
Hermes 4 405B,0.7939151676660005,/models/hermes-4-llama-3-1-405b/providers,false
Phi-4,0.7936447166921899,/models/phi-4/providers,false
o4-mini (high),0.7897368993259404,/models/o4-mini/providers,false
LFM2 1.2B,0.7897212543554007,/models/lfm2-1-2b/providers,false
Cogito v2.1,0.7883040935672515,/models/cogito-v2-1-reasoning/providers,false
Qwen3 Coder 30B A3B,0.7875098193244304,/models/qwen3-coder-30b-a3b-instruct/providers,false
Nova 2.0 Pro Preview,0.7872424722662441,/models/nova-2-0-pro/providers,false
Llama 4 Scout,0.7869235259778167,/models/llama-4-scout/providers,false
Grok Code Fast 1,0.786839266450917,/models/grok-code-fast-1/providers,false
Doubao Seed Code,0.7854640980735552,/models/doubao-seed-code/providers,false
Llama 3.1 70B,0.7799917830731307,/models/llama-3-1-instruct-70b/providers,false
GPT-5.2 (xhigh),0.7786302926967889,/models/gpt-5-2/providers,false
Ministral 3B (Dec '25),0.7780589192120008,/models/ministral-3b/providers,false
Mistral Small 3.1,0.7761135965765416,/models/mistral-small-3-1/providers,false
Nova Pro,0.7754532775453278,/models/nova-pro/providers,false
Qwen3 4B 2507,0.7753969772335948,/models/qwen3-4b-2507-instruct-reasoning/providers,false
GPT-5 (low),0.7742864624247185,/models/gpt-5-low/providers,false
Gemini 2.5 Flash-Lite,0.7737329042638778,/models/gemini-2-5-flash-lite-reasoning/providers,false
Qwen3 235B,0.7678137651821862,/models/qwen3-235b-a22b-instruct-reasoning/providers,false
Devstral Small,0.7660275033895022,/models/devstral-small/providers,false
Mistral Small 3.2,0.7647744945567652,/models/mistral-small-3-2/providers,false
Qwen3 235B 2507,0.7640040444893832,/models/qwen3-235b-a22b-instruct-2507/providers,false
Phi-4 Mini,0.7633670520231214,/models/phi-4-mini/providers,false
Command A,0.7611852433281004,/models/command-a/providers,false
Llama Nemotron Super 49B v1.5,0.7572989076464747,/models/llama-nemotron-super-49b-v1-5-reasoning/providers,false
K2-V2 (low),0.7558915946582875,/models/k2-v2-low/providers,false
Kimi K2,0.7535938903863432,/models/kimi-k2/providers,false
Qwen3 14B,0.7509221510386332,/models/qwen3-14b-instruct-reasoning/providers,false
Qwen3 4B 2507,0.7454614220877458,/models/qwen3-4b-2507-instruct/providers,false
Kimi K2 Thinking,0.7439943476212906,/models/kimi-k2-thinking/providers,false
GPT-5 Codex (high),0.7435082140964494,/models/gpt-5-codex/providers,false
Gemini 2.5 Flash,0.7423435419440746,/models/gemini-2-5-flash-reasoning/providers,false
Claude Opus 4.5,0.7422258592471358,/models/claude-opus-4-5/providers,false
Jamba Reasoning 3B,0.7421540656205421,/models/jamba-reasoning-3b/providers,false
NVIDIA Nemotron Nano 9B V2,0.7411139611579333,/models/nvidia-nemotron-nano-9b-v2/providers,false
DeepSeek V3.1 Terminus,0.7395881006864988,/models/deepseek-v3-1-terminus-reasoning/providers,false
Llama 3.3 Nemotron Super 49B,0.7341626794258374,/models/llama-3-3-nemotron-super-49b/providers,false
GPT-5.1 Codex (high),0.7312045270816492,/models/gpt-5-1-codex/providers,false
MiMo-V2-Flash,0.7279426409081856,/models/mimo-v2-flash/providers,false
Grok 4.1 Fast,0.7174291938997821,/models/grok-4-1-fast-reasoning/providers,false
Nova Premier,0.707261880271549,/models/nova-premier/providers,false
Granite 4.0 350M,0.7006060606060606,/models/granite-4-0-350m/providers,false
GLM-4.5,0.694614711033275,/models/glm-4.5/providers,false
Ring-1T,0.6923076923076923,/models/ring-1t/providers,false
Kimi K2 0905,0.6895568231680561,/models/kimi-k2-0905/providers,false
Llama 3.1 Nemotron 70B,0.6888933121019108,/models/llama-3-1-nemotron-instruct-70b/providers,false
Grok 4 Fast,0.6737922188969645,/models/grok-4-fast-reasoning/providers,false
GLM-4.6,0.6711956521739131,/models/glm-4-6/providers,false
ERNIE 4.5 300B A47B,0.665314401622718,/models/ernie-4-5-300b-a47b/providers,false
GLM-4.6V,0.6638,/models/glm-4-6v/providers,false
Llama 3.2 1B,0.6635747413485551,/models/llama-3-2-instruct-1b/providers,false
Llama Nemotron Super 49B v1.5,0.6632768361581921,/models/llama-nemotron-super-49b-v1-5/providers,false
KAT-Coder-Pro V1,0.6623058053965658,/models/kat-coder-pro-v1/providers,false
Gemini 2.5 Flash-Lite (Sep),0.6585881900365455,/models/gemini-2-5-flash-lite-preview-09-2025/providers,false
Nova Micro,0.6478484737035675,/models/nova-micro/providers,false
Grok 4,0.638996138996139,/models/grok-4/providers,false
Devstral Medium,0.6228105906313646,/models/devstral-medium/providers,false
NVIDIA Nemotron Nano 9B V2,0.6043689320388349,/models/nvidia-nemotron-nano-9b-v2-reasoning/providers,false
Mistral Medium 3,0.6035872632003224,/models/mistral-medium-3/providers,false
GPT-5.2,0.6016655100624566,/models/gpt-5-2-non-reasoning/providers,false
Mistral Large 2 (Nov),0.6007696981972858,/models/mistral-large-2/providers,false
Magistral Medium 1.2,0.5970802919708029,/models/magistral-medium-2509/providers,false
GPT-5 nano (high),0.5865796451152355,/models/gpt-5-nano/providers,false
Nova Lite,0.5829810696563131,/models/nova-lite/providers,false
Claude Opus 4.5,0.5780837972458248,/models/claude-opus-4-5-thinking/providers,false
GPT-5 mini (high),0.5527909995672868,/models/gpt-5-mini/providers,false
GPT-5 nano (medium),0.523021726131154,/models/gpt-5-nano-medium/providers,false
Claude 3.7 Sonnet,0.5199726089933805,/models/claude-3-7-sonnet/providers,false
Claude 4.5 Sonnet,0.5143704379562044,/models/claude-4-5-sonnet/providers,false
Llama 3.1 405B,0.511727078891258,/models/llama-3-1-instruct-405b/providers,false
GPT-5.1 (high),0.5115919629057187,/models/gpt-5-1/providers,false
GPT-5.1 Codex mini (high),0.5112862010221465,/models/gpt-5-1-codex-mini/providers,false
GLM-4.6V,0.48879716981132076,/models/glm-4-6v-reasoning/providers,false
Claude 4.1 Opus,0.4838709677419355,/models/claude-4-1-opus-thinking/providers,false
Claude 4.5 Sonnet,0.47732754462132176,/models/claude-4-5-sonnet-thinking/providers,false
GPT-5 mini (medium),0.43377063055438003,/models/gpt-5-mini-medium/providers,false
Claude 4 Sonnet,0.40504986208359856,/models/claude-4-sonnet/providers,false
Claude 3.7 Sonnet,0.3891670459717797,/models/claude-3-7-sonnet-thinking/providers,false
Ring-flash-2.0,0.3870274144418563,/models/ring-flash-2-0/providers,false
GPT-4o (Nov),0.37766393442622953,/models/gpt-4o/providers,false
Claude 4 Sonnet,0.28900147772852014,/models/claude-4-sonnet-thinking/providers,false
Claude 4.5 Haiku,0.2606880095446411,/models/claude-4-5-haiku-reasoning/providers,false
Grok 3 mini Reasoning (high),0.25321637426900584,/models/grok-3-mini-reasoning/providers,false
Claude 4.5 Haiku,0.2467757459095284,/models/claude-4-5-haiku/providers,false"""

    # Parse CSV
    csv_reader = csv.DictReader(io.StringIO(csv_raw))
    csv_rows = list(csv_reader)
    
    updates = {}
    for row in csv_rows:
        name = row['modelName']
        url = row['detailsUrl']
        rate = float(row['omniscienceHallucinationRate'])
        
        # Create a key that captures 'Thinking' or 'Reasoning' intent from the URL if needed
        is_thinking = "thinking" in url or "reasoning" in url
        updates[(name, is_thinking)] = rate

    # Logic to match and update models.json
    final_models = models_data.get('models', [])
    updated_count = 0
    
    # Track which CSV entries were used
    used_updates = set()

    print(f"DEBUG: Processing {len(final_models)} existing models...")

    for model in final_models:
        display_name = model.get('display_name', '')
        openrouter_id = model.get('openrouter_id', '')
        
        # Strategy: Match by display name or openrouter_id
        is_reasoning = "Reasoning" in display_name or ":thinking" in openrouter_id or "(high)" in display_name
        
        best_match = None
        for (csv_name, csv_thinking), rate in updates.items():
            # Normalized comparison
            if csv_name.lower().replace(" ", "") in display_name.lower().replace(" ", ""):
                # Special cases for Sonnet/Haiku to avoid cross-matching
                if "sonnet" in csv_name.lower() and "sonnet" not in display_name.lower() and "sonnet" not in openrouter_id.lower():
                    continue
                if "haiku" in csv_name.lower() and "haiku" not in display_name.lower() and "haiku" not in openrouter_id.lower():
                    continue
                
                # If both are reasoning or both are not
                if is_reasoning == csv_thinking:
                    best_match = rate
                    used_updates.add((csv_name, csv_thinking))
                    break
        
        if best_match is not None:
            # ONLY update hallucination_rate, never touch 'hle' (Humanity's Last Exam)
            model['hallucination_rate'] = round(best_match * 100, 2)
            updated_count += 1

    print(f"DEBUG: Updated {updated_count} existing models.")
    print(f"DEBUG: Processing remaining {len(updates) - len(used_updates)} CSV entries for additions...")

    # Add missing models (like Claude 3.7 Sonnet) from cache if they were in the CSV but not in models.json
    for (csv_name, csv_thinking), rate in updates.items():
        if (csv_name, csv_thinking) not in used_updates:
            # Try to find in cache
            found_in_cache = None
            for cache_model in cache_data.get('models', []):
                c_name = cache_model.get('display_name', cache_model.get('name', ''))
                c_id = cache_model.get('openrouter_id', '')
                c_is_reasoning = "Reasoning" in c_name or ":thinking" in c_id or "(high)" in c_name
                
                if csv_name.lower().replace(" ", "") in c_name.lower().replace(" ", ""):
                    # Special cases for Sonnet/Haiku
                    if "sonnet" in csv_name.lower() and "sonnet" not in c_name.lower() and "sonnet" not in c_id.lower():
                        continue
                    if "haiku" in csv_name.lower() and "haiku" not in c_name.lower() and "haiku" not in c_id.lower():
                        continue

                    if c_is_reasoning == csv_thinking:
                        found_in_cache = cache_model
                        break
            
            if found_in_cache:
                new_model = found_in_cache.copy()
                # Ensure HLE Reasoning score is preserved from cache
                # And set hallucination_rate from AA data
                new_model['hallucination_rate'] = round(rate * 100, 2)
                
                # provide defaults for missing fields if needed
                if 'description' not in new_model: new_model['description'] = ""
                if 'tags' not in new_model: new_model['tags'] = []
                
                # Map cost fields if they are different in cache
                if 'price_1m_input' in new_model and 'input_cost_per_m' not in new_model:
                    new_model['input_cost_per_m'] = new_model['price_1m_input']
                if 'price_1m_output' in new_model and 'output_cost_per_m' not in new_model:
                    new_model['output_cost_per_m'] = new_model['price_1m_output']
                
                final_models.append(new_model)
                updated_count += 1
                print(f"Added new model: {new_model['display_name']} with HLE {rate}")
            else:
                print(f"WARNING: Could not find '{csv_name}' (thinking={csv_thinking}) in cache.")

    # Save results
    with open(models_path, "w") as f:
        json.dump(models_data, f, indent=2)
    
    print(f"Successfully processed {updated_count} models in {models_path}")

if __name__ == "__main__":
    update_hallucination_rates()
