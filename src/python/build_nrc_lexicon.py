import requests
import csv
import io
import unicodedata

url = 'https://raw.githubusercontent.com/jboscomendoza/lexicos-nrc-afinn/master/lexico_nrc.csv'
r = requests.get(url)
content = r.content.decode('latin-1')

def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn').lower().strip()

# Emociones estándar normalizadas sin tildes:
# ira, asco, miedo, tristeza, sorpresa, anticipacion, alegria, confianza, positivo, negativo
emo_order = ['ira', 'asco', 'miedo', 'tristeza', 'sorpresa', 'anticipacion', 'alegria', 'confianza']

word_dict = {}

reader = csv.reader(io.StringIO(content))
header = next(reader)

for row in reader:
    if len(row) < 3:
        continue
    es_word = row[0].strip().lower()
    raw_emo = row[1].strip()
    en_word = row[2].strip().lower()

    norm_emo = strip_accents(raw_emo)
    if norm_emo == 'enfado':
        norm_emo = 'ira'
    elif 'alegr' in norm_emo:
        norm_emo = 'alegria'
    elif 'anticip' in norm_emo:
        norm_emo = 'anticipacion'

    for w in [es_word, en_word]:
        if not w:
            continue
        if w not in word_dict:
            word_dict[w] = set()
        word_dict[w].add(norm_emo)

out_file = 'src/main/resources/lexicons/nrc_emolex.csv'

with open(out_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['palabra'] + emo_order + ['intensidad', 'polaridad'])
    for w, emos in sorted(word_dict.items()):
        vals = [1 if e in emos else 0 for e in emo_order]
        count = sum(vals)
        intensity = min(1.0, 0.35 + 0.15 * count) if count > 0 else 0.5

        pos = 1 if 'positivo' in emos or ('alegria' in emos or 'confianza' in emos or 'anticipacion' in emos) else 0
        neg = 1 if 'negativo' in emos or ('ira' in emos or 'asco' in emos or 'miedo' in emos or 'tristeza' in emos) else 0
        polarity = 1 if (pos and not neg) else (-1 if (neg and not pos) else 0)

        writer.writerow([w] + vals + [f'{intensity:.2f}', polarity])

print(f"Léxico regenerado correctamente en {out_file} con {len(word_dict)} entradas.")
