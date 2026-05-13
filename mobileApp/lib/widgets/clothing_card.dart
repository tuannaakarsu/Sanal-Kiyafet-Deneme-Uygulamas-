import 'dart:io';
import 'package:flutter/material.dart';
import 'package:my_flutter_app/features/model/clothing_item_model.dart';

/// Tek bir kıyafeti temsil eden kart widget.
/// Görsel, isim ve kategori gösterir. Opsiyonel silme butonu içerir.
class ClothingCard extends StatelessWidget {
  final ClothingItem item;
  final VoidCallback? onDelete;

  const ClothingCard({
    super.key,
    required this.item,
    this.onDelete,
});

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Card(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          elevation: 3,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Üstte kıyafet görseli (veya ikon)
              Expanded(
                  child: item.imagePath.isNotEmpty
                      ? ClipRRect(
                          borderRadius: 
                            const BorderRadius.vertical(top: Radius.circular(12)),
                          child: Image.file(
                            File(item.imagePath),
                            height: 150,
                            fit: BoxFit.cover,
                          ),
                      )
                      : const Center(
                          child: Icon(Icons.image, size: 60),
                  )
              ),

              // Alt kısımda isim ve kategori
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(item.name, style: const TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 4),
                    Text(item.category, style: const TextStyle(color: Colors.grey)),
                  ],
                ),
              )
            ],
          ),
        ),

        // Opsiyonel: sağ üstte silme butonu
        if (onDelete != null)
        Positioned(
            top: 4,
            right: 4,
            child: IconButton(
                onPressed: onDelete,
                icon: const Icon(Icons.delete, color: Colors.redAccent, size: 35),
                tooltip: 'Sil',
            )
        )
      ],
    );
  }
}