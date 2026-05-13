/// Kıyafet nesnesini temsil eden model sınıfıdır.
/// Veritabanı işlemleri (insert/select) için kullanılır.
class ClothingItem {
  int? id;
  final String name;
  final String category;
  final String imagePath;

  ClothingItem({
    this.id,
    required this.name,
    required this.category,
    required this.imagePath,
  });

  /// Kıyafet verisini Map'e çevirir.
  /// Veritabanına yazmak için kullanılır.
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'category': category,
      'imagePath': imagePath,
    };
  }

  /// Map verisini ClothingItem nesnesine çevirir.
  /// Veritabanından okurken kullanılır.
  factory ClothingItem.fromMap(Map<String, dynamic> map) {
    return ClothingItem(
      id: map['id'] as int?,
      name: map['name'] as String,
      category: map['category'] as String,
      imagePath: map['imagePath'] as String,
    );
  }
}