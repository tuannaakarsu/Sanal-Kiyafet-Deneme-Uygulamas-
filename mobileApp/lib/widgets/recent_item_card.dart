import 'dart:io';
import 'package:flutter/material.dart';
import 'package:my_flutter_app/features/model/clothing_item_model.dart';

/// Ana sayfada kullanılan yatay kıyafet kartı widget'ı.
/// Sadece görsel gösterir, kıyafet ismi ve kategori içermez.
class RecentItemCard extends StatelessWidget {
  final ClothingItem item;

  const RecentItemCard({super.key, required this.item});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 140,
      margin: const EdgeInsets.symmetric(horizontal: 8),
      decoration: BoxDecoration(
        color: Colors.grey[200],
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.shade300.withAlpha((255 * 0.4).toInt()),
            blurRadius: 10,
            spreadRadius: 2,
            offset: const Offset(2, 4),
          )
        ]
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: item.imagePath.isNotEmpty
          ? Image.file(File(item.imagePath), fit: BoxFit.cover)
          : const Icon(Icons.image_not_supported,  size: 50),
      ),
    );
  }
}