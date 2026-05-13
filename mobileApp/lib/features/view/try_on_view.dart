import 'dart:io';
import 'package:flutter/material.dart';
import 'package:my_flutter_app/features/view/closet_view.dart';
import 'package:my_flutter_app/features/view/result_view.dart';
import 'package:provider/provider.dart';
import 'package:my_flutter_app/features/viewmodel/settings_viewmodel.dart';
import 'package:my_flutter_app/features/viewmodel/try_on_viewmodel.dart';
import 'package:my_flutter_app/core/services/api_service.dart';

/// Kullanıcının kendi fotoğrafıyla kıyafet denemesi yapabileceği ekran.
/// Kullanıcı kıyafeti kameradan çekebilir veya dolaptan seçebilir.
class TryOnView extends StatelessWidget {
  const TryOnView({super.key});

    @override
    Widget build(BuildContext context) {
      final userImage = Provider.of<SettingsViewModel>(context).userPhotoPath;
      final double imageWidth = MediaQuery.of(context).size.width * 0.5;

      return ChangeNotifierProvider(
        create: (_) => TryOnViewModel(),
        child: Consumer<TryOnViewModel>(
          builder: (context, viewModel, _) {
            return Scaffold(
              appBar: AppBar(title: const Text('Kıyafet Deneme'), centerTitle: true),
              body: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    /// Kullanıcı Fotoğrafı
                    if (userImage != null)
                      ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: Image.file(
                          File(userImage),
                          width: imageWidth,
                          height: 250,
                          fit: BoxFit.cover,
                        ),
                      )
                    else
                      const Text('Lütfen ayarlardan kullanıcı fotoğrafı ekleyin'),

                    const SizedBox(height: 16),

                    /// Seçilen Kıyafet Görseli
                    if (viewModel.hasClothing)
                      ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: Image.file(
                          viewModel.clothingImage!,
                          width: imageWidth,
                          height: 250,
                          fit: BoxFit.cover,
                        ),
                      )
                    else
                      Container(
                        height: 250,
                        width: 200,
                        decoration: BoxDecoration(
                          color: Colors.grey[300],
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Center(child: Text('Henüz kıyafet seçilmedi')),
                      ),

                    const SizedBox(height: 16),

                    /// Kıyafet ekleme butonları (kamera & dolap)
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton.icon(
                            icon: const Icon(Icons.photo_camera),
                            onPressed: viewModel.isLoading
                                ? null
                                : () => viewModel.captureClothingFromCamera(),
                            label: const Text('Kıyafet Çek')
                        ),
                        ElevatedButton.icon(
                            icon: const Icon(Icons.checkroom),
                            onPressed: viewModel.isLoading
                                ? null
                                : () async {
                              //Dolaba yönlendir
                              final selectedImagePath = await Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                      builder: (_) =>
                                        const ClosetView(isSelectionMode: true)
                                  )
                              );
                              if (selectedImagePath != null && context.mounted) {
                                viewModel.setClothingFromCloset(selectedImagePath);
                              }
                            },
                            label: const Text('Dolaptan Seç')
                        )
                      ],
                    ),

                    const SizedBox(height: 20),

                    /// Deneme işlemini başlatma butonu
                    viewModel.isLoading
                      ? const CircularProgressIndicator()
                      : ElevatedButton.icon(
                      icon: const Icon(Icons.play_arrow),
                      onPressed: () async {
                        final apiService = ApiService();

                        if (userImage != null && viewModel.hasClothing) {
                          viewModel.setLoading(true);

                          final resultFile = await apiService.sendImagesToPython(
                              userImage: File(userImage),
                              clothingImage: File(viewModel.selectedClothingPath!)
                          );

                          viewModel.setLoading(false);

                          if (resultFile != null && context.mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                    content: Text(
                                        'Deneme işlemi başarıyla tamamlandı!'))
                            );

                            // ResultView'a yönlendir
                            Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (context) =>
                                        ResultView(imagePath: resultFile.path),
                                )
                            );
                          } else {
                            ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                    content: Text(
                                        'Deneme işlemi başarısız oldu.'))
                            );
                          }
                        } else {
                          ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                  content: Text(
                                  'Lütfen kullanıcı ve kıyafet fotoğrafı seçin'))
                          );
                        }
                      },
                      label: const Text('Dene!'),
                      style: ElevatedButton.styleFrom(
                          foregroundColor: Colors.white,
                          backgroundColor: Colors.blue,
                          padding: const EdgeInsets.symmetric(
                              horizontal: 30, vertical: 12)
                      ),
                    )
                  ],
                ),
              ),
            );
          },
        )
      );
    }
  }