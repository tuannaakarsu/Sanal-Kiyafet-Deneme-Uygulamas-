import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

/// Kıyafet deneme ekranı için ViewModel.
/// Kullanıcının seçtiği kıyafeti ve işlemleri yönetir.
class TryOnViewModel extends ChangeNotifier {
  String? selectedClothingPath;
  bool isLoading = false;

  /// Yükleme durumunu ayarlar.
  void setLoading(bool value) {
    isLoading = value;
    notifyListeners();
  }

  /// Kameradan yeni bir kıyafet fotoğrafı çeker.
  Future<void> captureClothingFromCamera() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.camera);
    if (image != null) {
      selectedClothingPath = image.path;
      notifyListeners();
    }
  }

  /// Dolaptan seçilen kıyafetin yolunu alır.
  void setClothingFromCloset(String path) {
    selectedClothingPath = path;
    notifyListeners();
  }

  /// Dosya nesnesi olarak kıyafet görseli
  File? get clothingImage => selectedClothingPath != null
      ? File(selectedClothingPath!) : null;

  /// Seçim durumu
  bool get hasClothing => selectedClothingPath != null;
}