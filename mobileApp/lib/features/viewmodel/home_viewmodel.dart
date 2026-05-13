import 'package:flutter/material.dart';
import 'package:my_flutter_app/core/db/database_helper.dart';
import 'package:my_flutter_app/core/services/api_service.dart';
import 'package:my_flutter_app/features/model/clothing_item_model.dart';

/// Ana sayfa için ViewModel.
/// Son eklenen kıyafetleri yükler.
class HomeViewModel extends ChangeNotifier {
  List<ClothingItem> recentItems = [];

  List<Map<String, dynamic>> suggestions = [];
  final ApiService _apiService = ApiService();

  /// Vertabanından son eklenen kıyafetleri getirir.
  /// [limit] ile kaç adet kıyafet yükleneceği belirlenir
  Future<void> fetchRecentClothes({int limit = 5}) async {
    final items = await DatabaseHelper.instance.getAllClothingItems();

    // Listeyi ters çevirip son eklenenlerden ilk [limit] kadarını al
    recentItems = items.reversed.take(limit).toList();

    notifyListeners(); // Arayüzü güncelle
  }

  Future<void> fetchSuggestions() async {
    suggestions = await _apiService.sendClothingDataToServer();
    notifyListeners();
  }
}