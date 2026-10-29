# Miras Gezgini'ne tarama ekleme

Miras Gezgini, UNESCO Dünya Mirası listesindeki her alanı bir kürede gösterir. Bir alanın içinde yürünebilmesi için birinin onu taraması gerekiyor. Bu rehber, kendi çektiğin ya da açık lisanslı bulduğun bir taramayı siteye eklemeyi anlatıyor.

## 1. Çekmeden önce

- **İzinlere uy.** Birçok miras alanında drone uçurmak, tripod kullanmak ya da ticari çekim yapmak izne bağlı. Alanın kurallarını öğren ve uy; izinsiz drone görüntüsü kabul etmiyoruz.
- **İnsanlara saygı göster.** Kalabalık saatlerden kaçın. Yüzleri net seçilen kişiler varsa o kareleri çıkar ya da çekimi tekrarla.
- **Alanın hangisi olduğunu bul.** Küreden alanı seç; ayrıntı kartında UNESCO numarası yazar (ör. Efes: 1018). Bu numara kaydın anahtarı.

## 2. Çekim

- 360° kamera varsa: kamerayı başının üstünde, görünmez selfie çubuğuyla tut. Yoksa telefonla yatay çek.
- Yavaş yürü, sabit yükseklikte kal, aynı yerden iki farklı yönde geç. 1–3 dakika yeterli.
- Hareketli kalabalıklar, su yüzeyleri, camlar ve düz beyaz duvarlar sonucu bozar.

## 3. İşleme

[Colab not defterini](https://colab.research.google.com/github/Oguzhan61-dilmac/360-vid2splat/blob/main/notebooks/vid2splat_colab.ipynb) aç, `T4 GPU` seç ve hücreleri sırayla çalıştır. Son hücre `scene.spz` ve `scene.json` içeren bir zip indirir.

Başka bir araçla (Polycam, Scaniverse, Postshot, gsplat…) ürettiğin `.ply`, `.spz` ya da `.sog` dosyaları da olur; `scripts/prepare_trevi.py` hizalama ve ölçekleme için örnek.

## 4. Siteye ekleme

**Dosya 50 MB'tan küçükse:** depoyu fork'la, zip'i `web/scenes/<kısa-ad>/` klasörüne aç ve `web/data/scans.json` listesine bir kayıt ekle:

```json
{
  "id": "efes",
  "whc": 1018,
  "name": "Celsus Kütüphanesi",
  "site": "Efes",
  "country": "Türkiye",
  "year": 2015,
  "path": "scenes/efes/",
  "size": "24 MB",
  "sourceShort": "telefon videosundan, CC BY 4.0"
}
```

**Dosya büyükse:** Hugging Face'e (ücretsiz) yükle ve `path` yerine `"splat": "<dosya adresi>"` yaz. Hugging Face başka sitelerin dosyayı yüklemesine izin verdiği için görüntüleyici doğrudan oradan açar.

`scene.json` dosyasına bir `credit` alanı ekle; görüntüleyici bunu giriş kartında gösterir:

```json
"credit": { "title": "Celsus Kütüphanesi taraması", "authors": "Adın Soyadın", "url": "https://…", "license": "CC BY 4.0" }
```

Sonra bir pull request aç. Kod yazmak istemiyorsan [buradan bir istek aç](https://github.com/Oguzhan61-dilmac/360-vid2splat/issues/new?template=tarama.md); videonu ya da dosyanı bağlantı olarak eklemen yeterli.

## 5. Lisans

- Kendi çekimlerin için **CC BY 4.0** öneriyoruz: herkes kullanabilir, adın anılır.
- Başkasının taramasını ekliyorsan lisansın yeniden dağıtıma izin verdiğinden emin ol ve kaynağı `credit` alanına yaz. Lisansı belirsiz dosyaları eklemiyoruz.

## Geliştirme

Kodda değişiklik yaparken:

```bash
python -m http.server -d web     # http://localhost:8000
python scripts/build_sites.py    # miras listesini Wikidata'dan yeniden derler (birkaç dakika sürer)
```

İşlem hattına dokunan değişikliklerde sentetik testin sonucunu (`tests/evaluate.py`) pull request açıklamasına ekle.
