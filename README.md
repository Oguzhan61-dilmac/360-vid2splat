<p align="center">
  <img src="docs/banner.svg" alt="360-Vid2Splat: videodan, içinde yürünebilen 3D sahneye" width="100%">
</p>

<p align="center">
  <a href="https://oguzhan61-dilmac.github.io/360-vid2splat/"><img src="https://img.shields.io/badge/canl%C4%B1%20demo-a%C3%A7-e8b84a?style=flat-square" alt="Canlı demo"></a>
  <a href="https://colab.research.google.com/github/Oguzhan61-dilmac/360-vid2splat/blob/main/notebooks/vid2splat_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab'da aç"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/maliyet-0%20TL-2ea043?style=flat-square" alt="Maliyet: 0 TL">
  <a href="LICENSE"><img src="https://img.shields.io/badge/lisans-MIT-8c97a3?style=flat-square" alt="MIT lisansı"></a>
</p>

**360-Vid2Splat**, 360° kamerayla ya da telefonla çekilmiş bir videoyu tarayıcıda **WASD ile içinde yürünebilen** fotogerçekçi bir 3D sahneye çevirir. Arkada COLMAP ile kamera konumları bulunur, 3D Gaussian Splatting ile sahne öğrenilir ve sonuç tek dosyalık bir web görüntüleyicide açılır. Her şey ücretsiz Colab GPU'su ve GitHub Pages üzerinde çalışır.

> *English:* Turn a 360° or phone video into a photorealistic 3D scene you can walk through in the browser. Open-source pipeline (COLMAP + Gaussian Splatting) that runs end to end on free Colab GPUs and GitHub Pages.

<p align="center">
  <img src="docs/screenshot.jpg" alt="Görüntüleyicide 360° drone videosundan üretilmiş Big Sur sahnesi" width="100%">
  <br>
  <sub>Görüntüleyicide Big Sur sahnesi · 8K 360° drone videosundan üretilmiş splat, <a href="https://huggingface.co/datasets/Jamesbass/bigsur-360-colmap">Jamesbass/bigsur-360-colmap</a> (CC BY 4.0)</sub>
</p>

## Hemen dene

| | |
|---|---|
| **Oda** · Mip-NeRF 360 sahnesi, 51 MB | [oguzhan61-dilmac.github.io/360-vid2splat/#oda](https://oguzhan61-dilmac.github.io/360-vid2splat/#oda) |
| **Big Sur kıyısı** · 8K 360° drone videosundan, 133 MB | [oguzhan61-dilmac.github.io/360-vid2splat/#big-sur](https://oguzhan61-dilmac.github.io/360-vid2splat/#big-sur) |
| **Sentetik oda** · bizim hattın test sahnesi | [oguzhan61-dilmac.github.io/360-vid2splat/#sentetik-oda](https://oguzhan61-dilmac.github.io/360-vid2splat/#sentetik-oda) |
| **Kendi videon** · ücretsiz T4 GPU ile | [Colab not defterini aç](https://colab.research.google.com/github/Oguzhan61-dilmac/360-vid2splat/blob/main/notebooks/vid2splat_colab.ipynb) |

Elindeki herhangi bir `.ply`, `.spz`, `.splat`, `.ksplat` ya da `.sog` dosyasını görüntüleyiciye sürükleyip bırakarak da açabilirsin.

## Neler yapıyor

- **360° ve normal video:** Equirectangular 360° videoyu en-boy oranından tanır. 360 kameran yoksa telefon videosu ya da hazır bir görüntü klasörü de olur.
- **Çeken kişiyi siler:** 360 kamerada her karede görünen operatör, hem kamera konumu hesabında hem eğitimde maskelenir; sahnede hayalet figür kalmaz.
- **Gerçek ölçek:** Sahne yerçekimine hizalanır, zemin bulunur ve metreye ölçeklenir. Görüntüleyicide göz hizasında yürürsün.
- **Bulanık kareleri eler:** Video kısa dilimlere bölünür, her dilimden en net kare seçilir.
- **Web'e hazır çıktı:** Havada asılı "hayalet" Gaussian'lar temizlenir, dosya SPZ biçiminde ~10 kat sıkıştırılır.
- **Gezinti:** WASD ve fare, telefonda dokunmatik çubuk, çekim rotasında otomatik tur, rota haritası ve görünmez duvarlarla sahnenin boş yerlerine düşmeme.

## Nasıl çalışır

```mermaid
flowchart LR
    V[Video veya<br/>kare klasörü] --> K[En net kareler]
    K -->|360°| P[12 sanal kamera<br/>+ operatör maskesi]
    K -->|telefon| T[Tek kamera<br/>+ lens düzeltme]
    P --> S[COLMAP SfM<br/>rig kısıtlı]
    T --> S
    S --> H[Yerçekimi, zemin<br/>ve metrik ölçek]
    H --> G[gsplat eğitimi<br/>MCMC, maskeli kayıp]
    G --> C[Temizlik<br/>+ SPZ]
    C --> W[Web görüntüleyici<br/>Spark + three.js]
```

| Adım | Dosya | Ne yapar |
|---|---|---|
| Kare seçimi | [`frames.py`](vid2splat/frames.py) | Videoyu 0,5 sn'lik dilimlere böler, her dilimden Laplacian varyansı en yüksek kareyi alır. Netlik sahne içeriğine bağlı olduğu için her dilim sadece komşularıyla kıyaslanır. |
| 360° → sanal kameralar | [`sfm.py`](vid2splat/sfm.py) | Her panoramayı 4 yön × 3 eğimde 90°'lik 12 görüntüye böler ve COLMAP'te sabit bir rig olarak çözer. Kameranın altındaki 35°'lik koni maskelenir. |
| Hizalama | [`align.py`](vid2splat/align.py) | Ortalama "yukarı" yönünü +Y'ye çevirir, kameraların altındaki noktalardan zemini bulur, kamera-zemin mesafesini çekim yüksekliğine eşitleyerek ölçeği hesaplar. |
| Eğitim | [`train.py`](vid2splat/train.py) | gsplat'in kararlı API'siyle sade bir eğitici. MCMC yoğunlaştırma Gaussian sayısını sınırlar; operatör maskesi kayıptan çıkarılır. |
| Dışa aktarma | [`export.py`](vid2splat/export.py) | Şeffaf, aşırı büyük ve yalnız kalmış Gaussian'ları siler, PLY'yi SPZ'ye çevirir. |
| Görüntüleyici | [`web/index.html`](web/index.html) | Derleme gerektirmeyen tek dosya; GitHub Pages'te yayınlanır. |

## Kendi videonu işle (Colab, ücretsiz)

1. [Not defterini Colab'da aç](https://colab.research.google.com/github/Oguzhan61-dilmac/360-vid2splat/blob/main/notebooks/vid2splat_colab.ipynb) ve `Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU` seç.
2. Hücreleri sırayla çalıştır: kurulum → video yükle → kamera konumları → eğitim. Toplam yaklaşık 45–60 dakika.
3. Son hücrenin indirdiği zip'i `web/scenes/<sahne>/` klasörüne aç ve [`web/scenes/index.json`](web/scenes/index.json) dosyasına ekle:
   ```json
   {"id": "oda", "name": "Bizim oda", "path": "scenes/oda/"}
   ```
4. `main` dalına push et; GitHub Actions siteyi birkaç dakikada yeniden yayınlar.

360 kameran yoksa not defterindeki **"Örnek 360° veri indir"** hücresi, [Big Sur 360 veri setinin](https://huggingface.co/datasets/Jamesbass/bigsur-360-colmap) (CC BY 4.0) panoramalarını indirir.

### İyi çekim için

- **Yavaş yürü** (normal hızının yarısı) ve kamerayı hep aynı yükseklikte tut.
- 360 kamerayı **görünmez selfie çubuğuyla başının üstünde** tut; yatay sabitleme açık kalsın.
- Hareketli insanlardan, aynalardan, camlardan ve düz beyaz duvarlardan kaçın; doku, kamera konumu hesabının yakıtıdır.
- Aynı yerden iki farklı yönde geçmek (döngü kurmak) sonucu güçlendirir. 1–2 dakikalık video yeterli.
- 360 videoyu kameranın uygulamasından **equirectangular MP4** olarak dışa aktar (5.7K yeterli).

## Yerel kullanım

```bash
git clone https://github.com/Oguzhan61-dilmac/360-vid2splat.git && cd 360-vid2splat
pip install -r requirements.txt          # NVIDIA GPU varsa pycolmap yerine pycolmap-cuda12

# Kamera konumları + hizalama (CPU'da da çalışır)
python -m vid2splat sfm video.mp4 --out output/oda --capture-height 1.7

# Eğitim ve web'e dışa aktarma (NVIDIA GPU + Node.js gerekir)
pip install --no-build-isolation gsplat==1.5.3
python -m vid2splat train output/oda --steps 15000

# Görüntüleyiciyi aç
python -m http.server -d web            # http://localhost:8000
```

<details>
<summary><b>Tüm seçenekler</b></summary>

| Seçenek | Varsayılan | Açıklama |
|---|---|---|
| `--mode` | `auto` | `360`, `perspective` ya da en-boy oranından otomatik |
| `--capture-height` | 360: 1.7, telefon: 1.5 | Kameranın yerden yüksekliği (m); ölçek buradan gelir |
| `--interval` | 0.5 | Kaç saniyede bir kare seçileceği |
| `--every` | 1 | Girdi bir görüntü klasörüyse her kaçıncı görüntünün alınacağı |
| `--max-frames` | 200 | Kare sayısı üst sınırı |
| `--pano-width` | 3840 | 360 karelerin genişliği; sanal görüntüler bunun dörtte biri olur |
| `--nadir-mask` | 35 | 360'ta kameranın altında maskelenen koni (derece); 0 kapatır |
| `--layout` | `overlapping` | `cubemap` ile klasik küp haritası (karşılaştırma için) |
| `--steps` | 15000 | Eğitim adımı; 30000 en iyi kalite |
| `--max-splats` | 1000000 | Gaussian üst sınırı; web dosya boyutunu belirler |
| `--no-floaters` | | Dışa aktarmada hayalet temizliğini kapatır |
| `--cpu` | | GPU olsa bile COLMAP'i CPU'da çalıştırır |

</details>

## Görüntüleyici

| Masaüstü | Telefon | |
|---|---|---|
| `W` `A` `S` `D` / oklar | Sol alttaki çubuk | Yürü |
| Fare (tıklayınca imleç kilitlenir) | Ekranda sürükle | Etrafına bak |
| `Shift` | | Koş |
| `+` / `−` | | Hızı ayarla |
| `F` | Uçuş modu düğmesi | Serbest uçuş (`E` yüksel, `Q` alçal) |
| `T` | Otomatik tur düğmesi | Çekim rotasında gezdirir |
| `R` | Başa dön düğmesi | Başlangıç noktasına döner |

Adres çubuğuyla uzak bir sahne de açılabilir:

- `?splat=<dosya-adresi>&flip=1` tek bir splat dosyası açar. `flip=1`, COLMAP düzeninde gelen, yani ters görünen dosyalar içindir.
- `?scene=<klasör-adresi>` bu hattın ürettiği `scene.json` + `scene.spz` klasörünü açar.

Her iki durumda da dosyanın bulunduğu sunucunun CORS izni vermesi gerekir; Hugging Face veriyor.

## Test sonuçları

Hat, gerçek kamera konumları bilinen sentetik videolarla ölçüldü. [`make_synthetic.py`](tests/make_synthetic.py) dokulu bir oda içinde yürüyen kamera üretir; 360 modunda kameranın altında "çeken kişi" figürü de vardır. Ölçüm [`evaluate.py`](tests/evaluate.py) ile yapılır:

| Test | Konumlandırılan | Ölçek oranı (1 = doğru) | Yukarı yön hatası | Konum hatası (RMSE) |
|---|---|---|---|---|
| 360° video, 10 sn | 24/24 panorama | 0.997 | 0.10° | 0.3 mm |
| Telefon videosu, 10 sn | 24/24 kare | 0.998 | 0.15° | 1.1 mm |
| Big Sur, gerçek 360° drone verisi (20 panorama, CPU) | 20/20 panorama | ölçülmedi | ölçülmedi | 0.35 px yeniden izdüşüm |

Sentetik görüntülerde gürültü olmadığı için bu hatalar gerçek çekimlerdekinden çok düşüktür; amaç geometrinin, maskelemenin ve ölçek tahmininin doğru kurulduğunu göstermek. Eğitim kodu GPU'suz bir duman testinden ([`smoke_train_cpu.py`](tests/smoke_train_cpu.py)) geçiyor; GPU'daki ilk tam eğitim sonuçları buraya eklenecek.

```bash
python tests/make_synthetic.py output/syn360.mp4 --mode 360       # ~6 dk, CPU
python -m vid2splat sfm output/syn360.mp4 --out output/syn360 --capture-height 1.9 --interval 0.4
python tests/evaluate.py output/syn360.json output/syn360
```

## Proje yapısı

```text
360-vid2splat/
├── vid2splat/                 Python işlem hattı (python -m vid2splat)
│   ├── frames.py              en net kare seçimi
│   ├── sfm.py                 360 rig / telefon SfM (pycolmap)
│   ├── align.py               yerçekimi, zemin, metrik ölçek, scene.json
│   ├── train.py               gsplat MCMC eğitimi (maskeli)
│   ├── export.py              hayalet temizliği + SPZ
│   └── pipeline.py            adımları birleştirir
├── web/                       GitHub Pages'te yayınlanan görüntüleyici
│   ├── index.html
│   └── scenes/                galeri: index.json + sahne klasörleri
├── notebooks/                 ücretsiz Colab not defteri
├── tests/                     sentetik veri, doğruluk ölçümü, CPU duman testi
├── docs/                      README görselleri
└── .github/workflows/         Pages yayını
```

## Yol haritası

- [x] 360° ve telefon videosundan SfM, operatör maskesi, metrik hizalama
- [x] Ücretsiz Colab eğitimi, SPZ dışa aktarma, web görüntüleyici
- [x] Hazır görüntü klasörleri ve örnek veri setleriyle çalışma
- [ ] GPU'da tam eğitim sonuçlarının (PSNR, karşılaştırma görselleri) eklenmesi
- [ ] Hareketli insanları otomatik maskeleme (YOLO-seg / SAM 2)
- [ ] Çekim kalite raporu: "şu bölge az görüldü, tekrar çek" uyarıları
- [ ] Görüntüleyicide çarpışma (zemin ve duvar vokselleri) ve mesafe ölçme
- [ ] GPU'lu bir sunucu bulunduğunda "yükle, bekle, gez" sitesi (FastAPI + iş kuyruğu)

## Katkıda bulunma

Hata bulursan ya da bir fikrin varsa [issue aç](https://github.com/Oguzhan61-dilmac/360-vid2splat/issues). Kod katkıları için kendi dalında çalışıp pull request gönder; işlem hattına dokunan değişikliklerde sentetik testin (`tests/evaluate.py`) sonuçlarını PR açıklamasına eklemen yeterli.

## Teşekkürler

Bu proje şu açık kaynak çalışmaların üzerine kurulu: [COLMAP / pycolmap](https://colmap.github.io) (BSD), [gsplat](https://github.com/nerfstudio-project/gsplat) (Apache 2.0), [Spark](https://sparkjs.dev) (MIT), [three.js](https://threejs.org) (MIT) ve [splat-transform](https://github.com/playcanvas/splat-transform) (MIT). Orijinal INRIA 3DGS kodu ticari olmayan lisanslı olduğu için kullanılmadı.

Demo sahneleri: Big Sur, [Jamesbass/bigsur-360-colmap](https://huggingface.co/datasets/Jamesbass/bigsur-360-colmap) (CC BY 4.0); Oda, [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) veri setinden, [cakewalk/splat-data](https://huggingface.co/cakewalk/splat-data) üzerinden.

## Lisans

[MIT](LICENSE) © 2026 Oğuzhan Dilmaç
