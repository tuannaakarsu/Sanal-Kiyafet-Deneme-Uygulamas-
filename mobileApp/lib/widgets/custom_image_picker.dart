import 'dart:io';
import 'package:flutter/material.dart';

class CustomImagePicker extends StatelessWidget {
  final String? imagePath;
  final VoidCallback onCameraTap;
  final VoidCallback onGalleryTap;

  const CustomImagePicker({
    super.key,
    required this.imagePath,
    required this.onCameraTap,
    required this.onGalleryTap,
  });

  @override
  Widget build(BuildContext context) {
    DecorationImage? backgroundImage;

    if (imagePath != null) {
      final file = File(imagePath!);
      backgroundImage = DecorationImage(
        image: FileImage(file),
        fit: BoxFit.cover,
      );
    }

    return Container(
      height: 420,
      key: ValueKey(imagePath),
      decoration: BoxDecoration(
        color: Colors.grey[300],
        borderRadius: BorderRadius.circular(10),
        image: backgroundImage,
      ),
      child: Stack(
        children: [
          if (imagePath == null)
            const Center(
              child: Icon(Icons.image, size: 60, color: Colors.grey),
            ),
          Positioned(
            bottom: 8,
            left: 8,
            child: IconButton(
              onPressed: onGalleryTap,
              icon: const Icon(Icons.photo_library),
              iconSize: 40,
              tooltip: 'Galeriden Seç',
            ),
          ),
          Positioned(
            bottom: 8,
            right: 8,
            child: IconButton(
              onPressed: onCameraTap,
              icon: const Icon(Icons.photo_camera),
              iconSize: 40,
              tooltip: 'Kameradan Çek',
            ),
          ),
        ],
      ),
    );
  }
}
