# GAN Gecis ve Try-On Iyilestirme Plani

Bu plan mevcut projeyi "yalnizca UNet segmentation + basit warp" yapisindan
"pose-aware ve kiyafet uyumlu virtual try-on GAN" yapisina tasimak icin
uygulanabilir adimlari verir.

## 1) Hedef Mimari (2 Asamali)

1. Warping Agi (W-Net):
- Giris: `person_agnostic`, `cloth_image`, `cloth_mask`, `pose_map`
- Cikis: `warped_cloth`, `warped_mask`, opsiyonel `flow_field`

2. Render GAN (R-Net + D):
- Generator (R-Net) girisi:
  - `person_agnostic`
  - `warped_cloth`
  - `warped_mask`
  - `pose_map`
- Cikis: final try-on RGB goruntu
- Discriminator: PatchGAN (real/fake + local doku kontrolu)

Neden:
- Tek adimda sadece homography ile warp, kol-govde pozu degistiginde
  kiyafeti dogru oturtamiyor.
- 2 asamali yapi, once geometriyi (warp), sonra goruntu gercekciligini
  (render) ayri optimize eder.

## 2) Veri Hazirlama (Proje Icindeki Yapiya Gore)

Mevcut veri:
- `datasets/kiyafet/IMAGES`
- `datasets/kiyafet/MASKS`
- `datasets/kiyafet/labels.csv`

Eklenmesi gerekenler:
1. `datasets/kiyafet/POSE`:
- Her person image icin pose heatmap veya keypoint json.
- Ilk versiyon: MediaPipe ile keypoint cikartip heatmap'e cevir.

2. `datasets/kiyafet/AGNOSTIC`:
- Yuz/sac korunmus, govde bolgesi neutral hale getirilmis person temsili.
- Baslangicta parsing mask + dilate ile elde edilebilir.

3. Egitim csv:
- `datasets/kiyafet/train_pairs.csv`
- Kolonlar:
  - `person_image`
  - `cloth_image`
  - `cloth_mask`
  - `pose_map`
  - `agnostic_image`
  - `target_image`

Not:
- Elindeki veri dogrudan paired try-on degilse once "self-reconstruction"
  kurulumuyla basla: cloth ve target ayni item olabilir.

## 3) Loss Tasarimi

Toplam loss:

`L = lambda_adv * L_adv + lambda_l1 * L_l1 + lambda_perc * L_perc + lambda_edge * L_edge + lambda_mask * L_mask`

Onerilen baslangic agirliklari:
- `lambda_adv = 1.0`
- `lambda_l1 = 10.0`
- `lambda_perc = 5.0`
- `lambda_edge = 2.0`
- `lambda_mask = 5.0`

Aciklama:
- `L_adv`: gercekcilik ve tekstur
- `L_l1`: global tutarlilik
- `L_perc`: VGG feature benzerligi
- `L_edge`: kiyafet kenari ve dikiş benzeri detaylari korur
- `L_mask`: warp maskesini hedef kiyafet bolgesine hizalar

## 4) Egitim Stratejisi

Faz A (Stabil baslangic):
1. Warping agini tek basina pretrain et (`L_l1 + L_mask + L_edge`)
2. Render generator'u discriminator olmadan pretrain et (`L_l1 + L_perc`)

Faz B (GAN ince ayar):
1. PatchGAN discriminator'u ac
2. Kademeli olarak `lambda_adv` arttir (0.1 -> 0.5 -> 1.0)

Faz C (Hard-case fine-tune):
1. Pozu zor ornekler (kol capraz, yandan durus)
2. Oversize / loose kiyafet alt veri grubu

## 5) Metrikler

Her epoch sonunda kaydet:
- `LPIPS` (dusuk daha iyi)
- `SSIM` (yuksek daha iyi)
- `Edge IoU` (kiyafet siniri)
- `Pose Alignment Error` (keypoint bolgelerinde kayma)

Ek:
- `logs/gan_log_*.csv` benzeri bir log yapisi kullan.

## 6) Kod Entegrasyonu

Bu repo icin onerilen dosyalar:

1. `train_gan_pose_tryon.py`:
- Baslangic GAN egitim scripti (ekledim)

2. `segmentasyon.py`:
- Kiyafet maskesi cikarma kalitesi artir (dominant class yerine
  daha kararlı class/pixel secimi)

3. `kıyafetigiydir.py`:
- Homography fallback olarak kalsin
- Yeni pipeline aktifken `warped_cloth` GAN’den gelsin

4. `anakod.py`:
- "legacy warp" ve "gan tryon" secici flag eklenebilir

## 7) Hizli Yol Haritasi (7 Gun)

Gun 1:
- Pose ve agnostic veri uretim scripti

Gun 2:
- `train_pairs.csv` olusturma + DataLoader test

Gun 3:
- Warping pretrain

Gun 4:
- Render pretrain

Gun 5:
- GAN fine-tune

Gun 6:
- Metrik ve ornek goruntu kaydetme

Gun 7:
- Mevcut Flask endpoint'e yeni modeli baglama

## 8) Kritik Hatalar ve Cozumler

1. Kiyafet cikarimi eksik:
- Mask quality dusuk -> cloth segmentation iyilestir + edge loss arttir.

2. Pose ustune oturmama:
- Sadece 4 nokta warp yetersiz -> flow/TPS warping ekle.

3. Kol ve kiyafet cakisma artefakti:
- Occlusion mask ve person-agnostic girdi kullan.

4. GAN egitimi unstable:
- Once adversarial kapali pretrain, sonra kademeli ac.

