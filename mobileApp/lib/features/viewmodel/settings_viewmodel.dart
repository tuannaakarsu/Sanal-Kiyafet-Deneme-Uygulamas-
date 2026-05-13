import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Uygulamanın ayarlarını yönetir.
/// - Kullanıcı fotoğrafını saklar
/// - Karanlık mod ayarını yönetir
class SettingsViewModel extends ChangeNotifier {
  String? userPhotoPath;
  bool isDarkMode = false;

  /// SharedPreferences'dan daha önce kaydedilen verileri yükler.
  Future<void> loadPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    userPhotoPath = prefs.getString('userPhoto');
    isDarkMode = prefs.getBool('darkMode') ?? false;
    notifyListeners();
  }

  /// Kullanıcı fotoğrafı yolunu kaydeder ve günceller.
  Future<void> setUserPhoto(String path) async {
    userPhotoPath = path;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('userPhoto', path);
    notifyListeners();
  }

  /// Karanlık modu aç/kapat ve kaydet.
  Future<void> toggleDarkMode(bool value) async {
    isDarkMode = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('darkMode', value);
    notifyListeners();
  }
}