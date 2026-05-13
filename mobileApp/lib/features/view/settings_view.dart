import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:my_flutter_app/features/viewmodel/settings_viewmodel.dart';

/// Ayarlar sayfası.
/// - Kullanıcı fotoğrafını yükleyebilir.
/// - Karanlık tema modu açılıp kapatılabilir.
class SettingsView extends StatelessWidget {
  const SettingsView({super.key});

  @override
  Widget build(BuildContext context) {
    final viewModel = Provider.of<SettingsViewModel>(context);

    return Scaffold(
      appBar: AppBar(title: const Text('Ayarlar')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text(
            'Kullanıcı Fotoğrafı',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),),
          const SizedBox(height: 15),

          /// Kullanıcı Fotoğrafı Seçme Alanı
          Center(
            child: GestureDetector(
              onTap: () async {
                final ImagePicker picker = ImagePicker();
                final XFile? image = await picker.pickImage(source: ImageSource.gallery);
                if (image != null) {
                  viewModel.setUserPhoto(image.path);
                }
              },
              child: ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: Container(
                  width: 250,
                  height: 300,
                  color: Colors.grey.shade300,
                  child: viewModel.userPhotoPath != null
                    ? Image.file(
                      File(viewModel.userPhotoPath!),
                      fit: BoxFit.cover,
                      )
                    : const Center(
                    child: Icon(Icons.add_a_photo, size: 32),
                  )
                ),
              ),
            ),
          ),
          const SizedBox(height: 24),

          /// Karanlık Tema Ayarı
          SwitchListTile(
              title: const Text('Karanlık Tema'),
              value: viewModel.isDarkMode,
              onChanged: (value) {
                viewModel.toggleDarkMode(value);
              }
          )
        ],
      ),
    );
  }
}