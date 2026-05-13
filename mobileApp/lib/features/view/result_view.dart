import 'dart:io';
import 'package:flutter/material.dart';

class ResultView extends StatelessWidget {
  final String imagePath;

  const ResultView({super.key, required this.imagePath});

  @override
  Widget build(BuildContext context) {
    final imageFile = File(imagePath);

    return Scaffold(
      appBar: AppBar(title: const Text("Deneme Sonucu"), centerTitle: true),
      body: imageFile.existsSync()
        ? Column(
        children: [
          Expanded(
            child: Image.file(
                imageFile,
                fit: BoxFit.cover,
                gaplessPlayback: true,
                key: ValueKey('${imageFile.path}_${imageFile.lastModifiedSync()}')
            ),
          ),
          const SizedBox(height: 50),
        ],
      )
      : const Center(
        child: Text(
          "Görsel bulunamadı",
          style: TextStyle(fontSize: 16, color: Colors.grey),
        ),
      )
    );
  }
}