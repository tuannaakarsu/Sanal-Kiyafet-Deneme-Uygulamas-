import 'package:flutter/material.dart';
import 'package:my_flutter_app/features/view/add_clothing_view.dart';

class SuggestionCard extends StatelessWidget {
  final Map<String, dynamic> suggestion;

  const SuggestionCard({super.key, required this.suggestion});

  @override
  Widget build(BuildContext context) {
    final String imageUrl = suggestion['image_path'] ?? '';

    return Material(
      elevation: 3,
      borderRadius: BorderRadius.circular(12),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () {
          // Kıyafet ekle sayfasına seçilen resimle yönlendir
          Navigator.push(
            context,
            MaterialPageRoute(
                builder: (_) => AddClothingView(imageUrl: imageUrl),
            )
          );
        },
        borderRadius: BorderRadius.circular(12),
        child: Stack(
          children: [
            imageUrl.isEmpty
                ? Container(
              color: Colors.grey[200],
              alignment: Alignment.center,
              child: const Icon(Icons.broken_image, size: 40, color: Colors.grey),
            )
                : Image.network(
              imageUrl,
              fit: BoxFit.cover,
              width: double.infinity,
              height: double.infinity,
              loadingBuilder: (context, child, progress) {
                if (progress == null) return child;
                return const Center(child: CircularProgressIndicator());
              },
              errorBuilder: (context, error, stackTrace) => Container(
                color: Colors.grey[200],
                alignment: Alignment.center,
                child: const Icon(Icons.error, color: Colors.red, size: 40),
              ),
            ),

            // Kıyafeti dolaba kaydetme butonu
            Positioned(
              bottom: 8,
              right: 8,
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(30),
                ),
                padding: const EdgeInsets.all(6.0),
                child: const Icon(Icons.add_box_rounded, color: Colors.white),
              ),
            ),
          ],
        ),
      )
    );
  }
}
