import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:my_flutter_app/features/viewmodel/main_viewmodel.dart';
import 'package:my_flutter_app/widgets/bottom_navbar.dart';

/// Uygulamanın ana yapısını yöneten View.
/// Seçilen sekmeye göre ilgili ekranı gösterir ve alt gezinme çubuğunu barındırır.
class MainView extends StatelessWidget {
  const MainView({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<MainViewModel>(
        builder: (context, viewModel, child) {
          return Scaffold(
            body: viewModel.currentPage, //Seçilen sayfayı göster
            bottomNavigationBar: BottomNavBar(
                currentIndex: viewModel.currentIndex,
                onTap: (index) => viewModel.setIndex(index), //Sayfa değiştir
            ),
          );
        }
    );
  }
}