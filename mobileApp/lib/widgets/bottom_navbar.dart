import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

/// Alt gezinme çubuğu bileşeni.
class BottomNavBar extends StatelessWidget {
  final int currentIndex;
  final ValueChanged<int> onTap;

  const BottomNavBar({
    super.key,
    required this.currentIndex,
    required this.onTap,
});

  @override
  Widget build(BuildContext context) {
    return BottomNavigationBar(
      currentIndex: currentIndex,
      onTap: onTap,
      selectedItemColor: Colors.blue,
      unselectedItemColor: Colors.grey,
      type: BottomNavigationBarType.fixed,
      items: const [
        BottomNavigationBarItem(
          icon: Icon(Icons.home),
          label: "Ana Sayfa",
        ),
        BottomNavigationBarItem(
          icon: Icon(FontAwesomeIcons.wandMagicSparkles),
          label: "Kıyafet Dene",
        ),
        BottomNavigationBarItem(
          icon: Icon(Icons.checkroom),
          label: "Dolabım",
        ),
      ],
    );
  }
}