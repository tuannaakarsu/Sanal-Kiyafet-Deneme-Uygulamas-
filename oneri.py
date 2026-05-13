import os
import json
from collections import Counter

def normalize_url_base(url_base: str) -> str:
    return url_base.rstrip("/") + "/"

def guncelle_katalog_json(ngrok_url):
    if not ngrok_url:
        print("Ngrok URL sağlanmadı.")
        return

    ngrok_url = normalize_url_base(ngrok_url)

    json_path = os.path.join(os.path.dirname(__file__), "tumkatalog.json")

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

        if not content:
            print("Hata: JSON dosyası boş.")
            return

        katalog = json.loads(content)

    except Exception as e:
        print("JSON okuma hatası:", e)
        return

    # 2. image_path alanlarını güncelle
    try:
        for urun in katalog:
            fotograf = urun.get("image_path","").strip()

            # Eğer URL http veya https ile başlıyorsa:
            if fotograf.startswith("http://") or fotograf.startswith("https://"):

                if not fotograf.startswith(ngrok_url):

                    filename = os.path.basename(fotograf)

                    urun["image_path"] = f"{ngrok_url}tumkatalog/{filename}"
                else:
                    # Aynı base URL ise olduğu gibi bırak
                    urun["image_path"] = fotograf

            else:
                filename = os.path.basename(fotograf)
                if "." not in filename:
                    filename += ".jpeg"
                urun["image_path"] = f"{ngrok_url}tumkatalog/{filename}"

    except Exception as e:
        print("Veri işleme hatası:", e)
        return

    # 3. Güncellenmiş JSON'u yaz
    try:
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(katalog, file, indent=4, ensure_ascii=False)
        print("Katalog JSON dosyası başarıyla güncellendi.")
    except Exception as e:
        print("JSON yazma hatası:", e)


def oneri_uret(kullanici_verisi, ngrok_url):
    if not kullanici_verisi:
        return []

    # Verilen ngrok_url ile fotoğraf linklerini güncelle
    guncelle_katalog_json(ngrok_url)

    json_path = os.path.join(os.path.dirname(__file__), "tumkatalog.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            katalog = json.load(f)
    except Exception as e:
        print("Katalog dosyası okunamadı:", e)
        return []

    kiyafet_ids = [u.get("id") for u in kullanici_verisi if "id" in u]
    kategoriler = [u.get("category") for u in kullanici_verisi if "category" in u]

    if not kategoriler:
        return []

    en_cok_kategori = Counter(kategoriler).most_common(1)[0][0]

    onerilen = [
        urun for urun in katalog
        if urun.get("category", "").lower() == en_cok_kategori.lower()
           and urun.get("id") not in kiyafet_ids
    ]

    return onerilen[:4]
