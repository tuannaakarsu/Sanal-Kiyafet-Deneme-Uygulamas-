import 'package:flutter/material.dart';
import 'package:my_flutter_app/core/db/database_helper.dart';
import 'package:my_flutter_app/features/model/clothing_item_model.dart';

/// ClosetViewModel, dolap sayfasının mantığını yönetir.
/// Kıyafetleri filtreleme, arama, silme işlemleri buradan kontrol edilir.
class ClosetViewModel extends ChangeNotifier {
  List<ClothingItem> _allClothes = [];
  List<ClothingItem> filteredClothes = [];

  String? selectedCategory;
  String searchQuery = '';

  /// Veritabanından kıyafetleri çeker ve filtre uygular.
  Future<void> fetchClothes() async {
    _allClothes = await DatabaseHelper.instance.getAllClothingItems();
    _applyFilters();
  }

  /// Filtreleme işlemini uygular:
  /// - Seçili kategoriye göre süzme
  /// - Arama sorgusuna göre süzme
  void _applyFilters() {
    filteredClothes = _allClothes.where((item) {
      final matchesCategory =
          selectedCategory == null || item.category == selectedCategory;
      final matchesSearch =
          item.name.toLowerCase().contains(searchQuery.toLowerCase());
      return matchesCategory && matchesSearch;
    }).toList();
    notifyListeners(); // UI'ı güncelle
  }

  /// Kategori filtresini ayarlar ve filtreleri yeniden uygular.
  void setCategory(String? category) {
    selectedCategory = category;
    _applyFilters();
  }

  /// Arama sorgusunu ayarlar ve filtreleri yeniden uygular.
  void setSearchQuery(String query) {
    searchQuery = query;
    _applyFilters();
  }

  /// Verilen ID'li kıyafeti siler ve listeyi günceller.
  Future<void> deleteClothing(int id) async {
    await DatabaseHelper.instance.deleteClothingItem(id);
    await fetchClothes(); // Silmeden sonra listeyi yenile
  }
}