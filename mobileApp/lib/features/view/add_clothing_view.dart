import 'package:flutter/material.dart';
import 'package:my_flutter_app/widgets/custom_image_picker.dart';
import 'package:provider/provider.dart';
import 'package:my_flutter_app/features/viewmodel/add_clothing_viewmodel.dart';

/// Kullanıcının yeni bir kıyafet ekleyebileceği sayfa.
/// Kıyafet adı, kategori ve görsel alınır, veritabanına kaydedilir.
class AddClothingView extends StatefulWidget {
  final String? imageUrl;

  const AddClothingView({super.key, this.imageUrl});

  @override
  State<AddClothingView> createState() => _AddClothingViewState();
}

class _AddClothingViewState extends State<AddClothingView> {
  late final AddClothingViewModel _viewModel;

    @override
    void initState() {
      super.initState();
      _viewModel = AddClothingViewModel();

      if (widget.imageUrl != null) {
        // Görsel verisini modelde başlat
        _viewModel.initializeWithSuggestion(widget.imageUrl);
      } else {
        _viewModel.isInitialized = true;
      }
    }

    @override
    void dispose() {
      _viewModel.disposeFields();
      super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider<AddClothingViewModel>.value(
      value: _viewModel,
      child: Consumer<AddClothingViewModel>(
        builder: (context, viewModel, _) {
          if (!viewModel.isInitialized) {
            return const Scaffold(
              body: Center(child: CircularProgressIndicator()),
            );
          }
          return Scaffold(
            resizeToAvoidBottomInset: true,
            appBar: AppBar(
              title: Text('Kıyafet Ekle'),
              centerTitle: true,
            ),
            body: SingleChildScrollView(
              padding: const EdgeInsets.all(18.0),
              child: Form(
                key: viewModel.formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Görsel önizleme ve seçim alanı
                    CustomImagePicker(
                      imagePath: viewModel.imagePath,
                      onCameraTap: () => viewModel.pickImage(fromCamera: true),
                      onGalleryTap: () => viewModel.pickImage(fromCamera: false),
                    ),

                    const SizedBox(height: 25),

                    /// Kıyafet Adı Alanı
                    TextFormField(
                      controller: viewModel.nameController,
                      decoration: const InputDecoration(
                        labelText: 'Kıyafet Adı',
                        border: OutlineInputBorder(),
                      ),
                      validator: (value) => value == null || value.isEmpty
                        ? 'Boş bırakılamaz'
                        : null,
                    ),

                    const SizedBox(height: 20),

                    /// Kategori Seçim Alanı
                    DropdownButtonFormField<String>(
                      decoration: const InputDecoration(
                        labelText: 'Kategori',
                        border: OutlineInputBorder(),
                      ),
                      value: viewModel.selectedCategory,
                      items: viewModel.categories
                          .map((category) => DropdownMenuItem(
                            value: category,
                            child: Text(category),
                          )).toList(),
                      onChanged: viewModel.setCategory,
                      validator: (value) =>
                        value == null ? 'Kategori seçiniz' : null,
                    ),

                    const SizedBox(height: 25),

                    /// Kaydet Butonu
                    ElevatedButton(
                      onPressed: () => viewModel.handleSave(context),
                      child: const Text('Kaydet'),
                    ),
                  ]
                ),
              ),
            )
          );
        },
      ),
    );
  }
}