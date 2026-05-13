import 'package:flutter/material.dart';
import 'package:my_flutter_app/features/view/home_view.dart';
import 'package:my_flutter_app/features/view/try_on_view.dart';
import 'package:my_flutter_app/features/view/closet_view.dart';

/// Alt gezinme çubuğundaki sekmeleri yöneten ViewModel.
/// Seçilen indekse göre ilgili sayfa gösterilir.
class MainViewModel extends ChangeNotifier {
  int _currentIndex = 0;
  final List<Widget> _pages = [
    HomeView(),
    TryOnView(),
    ClosetView(),
  ];

  int get currentIndex => _currentIndex;
  Widget get currentPage => _pages[_currentIndex];

  /// Yeni sekme seçildiğinde index'i günceller ve UI'ı yeniler
  void setIndex(int index) {
    _currentIndex = index;
    notifyListeners();
  }
}