import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:my_flutter_app/core/db/database_helper.dart';
import 'package:my_flutter_app/features/model/clothing_item_model.dart';
import 'package:path_provider/path_provider.dart';

/// Kıyafet ekleme sayfasının ViewModel'i.
/// Form durumunu ve kayıt işlemlerini yönetir.
class AddClothingViewModel extends ChangeNotifier {
  final formKey = GlobalKey<FormState>();
  final nameController = TextEditingController();
  String? selectedCategory;
  String? imagePath;
  bool isInitialized = false;

  final List<String> categories = ['Tişört', 'Pantolon', 'Ceket', 'Etek', 'Elbise'];

  /// Galeri veya kameradan resim seç
  Future<void> pickImage({required bool fromCamera}) async {
    final source = fromCamera ? ImageSource.camera : ImageSource.gallery;
    final pickedFile = await ImagePicker().pickImage(source: source);

    if (pickedFile != null) {
      imagePath = pickedFile.path;
      notifyListeners();
    }
  }

  /// Kıyafeti veritabanına kaydet
  Future<void> saveClothingItem(BuildContext context) async {
    final item = ClothingItem(
        name: nameController.text,
        category: selectedCategory ?? '',
        imagePath: imagePath ?? '',
    );

    await DatabaseHelper.instance.insertClothingItem(item);

    if (context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Kıyafet başarıyla eklendi!'))
      );
      Navigator.pop(context);
    }
  }

  void handleSave(BuildContext context) {
    if (formKey.currentState!.validate()) {
      if (imagePath == null) {
        showDialog(
            context: context,
            builder: (_) => AlertDialog(
              title: const Text('Eksik Bilgi'),
              content: const Text('Lütfen bir kıyafet fotoğrafı ekleyin.'),
              actions: [
                TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Tamam'),
                )
              ],
            )
        );
      } else {
        saveClothingItem(context);
      }
    }
  }

  void setCategory(String? value) {
    selectedCategory = value;
    notifyListeners();
  }

  /// Temizlik işlemi
  void disposeFields() {
    nameController.clear();
    selectedCategory = null;
    imagePath = null;
    isInitialized = false;
    notifyListeners();
  }

  Future<void> initializeWithSuggestion(String? url) async {
    if (isInitialized || url == null) return;

    try {
      isInitialized = false;
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final tempDir = await getTemporaryDirectory();
        final file = File(
            '${tempDir.path}/suggested_${DateTime
                .now()
                .millisecondsSinceEpoch}.jpg');
        await file.writeAsBytes(response.bodyBytes);
        imagePath = file.path;
        debugPrint('initializeWithSuggestion: imagePath set to $imagePath');
      }
    } catch (e) {
      debugPrint("URL'den görsel alınamadı: $e");
    } finally {
      isInitialized = true;
      notifyListeners();
    }
  }
}