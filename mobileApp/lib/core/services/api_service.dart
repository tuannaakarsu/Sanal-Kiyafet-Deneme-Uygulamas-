import 'dart:developer';
import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:my_flutter_app/core/db/database_helper.dart';
import 'package:my_flutter_app/features/model/clothing_item_model.dart';

/// API servis işlemlerinden sorumlu sınıf.
/// Görüntü gönderme, öneri alma gibi sunucu etkileşimlerini kapsar.
class ApiService {
  final String baseUrl = 'https://a52b-78-183-92-13.ngrok-free.app';

  /// Kullanıcı ve kıyafet görüntülerini Python sunucusuna gönderir.
  /// Başarılı olursa işlenmiş sonucu içeren bir dosya döner.
  Future<File?> sendImagesToPython({
    required File userImage,
    required File clothingImage,
  }) async {
    final uri = Uri.parse('$baseUrl/vton');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath(
        'user_image',
        userImage.path,
        contentType: MediaType('image', 'jpeg'),
      ))
      ..files.add(await http.MultipartFile.fromPath(
        'clothing_image',
        clothingImage.path,
        contentType: MediaType('image', 'jpeg'),
      ));

    try {
      final response = await request.send();

      if (response.statusCode == 200) {
        final bytes = await response.stream.toBytes();
        final timestamp = DateTime.now().millisecondsSinceEpoch;
        final tempFile = File('${Directory.systemTemp.path}/tryon_result_$timestamp.png');
        await tempFile.writeAsBytes(bytes);

        return tempFile;
      }
    } catch (e) {
      log('sendImagesToPython hatası: $e');
    }
    return null;
  }

  /// Veritabanındaki kıyafet verilerini JSON olarak sunucuya gönderir.
  /// Sunucudan dönen önerileri [Map<String, dynamic>] listesi olarak verir.
  Future<List<Map<String, dynamic>>> sendClothingDataToServer() async {
    try {
      final List<ClothingItem> clothes =
        await DatabaseHelper.instance.getAllClothingItems();

      // id ve category bilgilerini içeren JSON verisi oluştur
      final List<Map<String, dynamic>> clothingData = clothes
          .where((item) => item.id != null && item.category.isNotEmpty)
          .map((item) => {
            'id': item.id,
            'category': item.category
          }).toList();

      final url = Uri.parse('$baseUrl/vton');

      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'clothes': clothingData})
      );

      if (response.statusCode != 200) {
        log('Sunucu hatası: ${response.statusCode}');
        return [];
      }

      final responseData = jsonDecode(response.body);
      final suggestions = responseData['oneriler'];
      if (suggestions is List) {
        return suggestions.cast<Map<String, dynamic>>();
      }
    } catch (e) {
      log('sendClothingDataToServer hatası: $e');
    }
    return [];
  }
}