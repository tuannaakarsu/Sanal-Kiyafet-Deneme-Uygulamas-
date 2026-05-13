import 'dart:developer';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../../features/model/clothing_item_model.dart';

/// Veritabanı tablosu ve sütun adları için sabitler.
class DBConstants {
  static const String dbName = 'wardrobe.db';
  static const String tableClothes = 'clothes';
  static const String columnId = 'id';
  static const String columnName = 'name';
  static const String columnCategory = 'category';
  static const String columnImagePath = 'imagePath';
}

/// SQLite veritabanı işlemleri için yardımcı sınıf.
/// Singleton tasarımı kullanılarak tek bir örnekle tüm uygulama boyunca işlem yapılır.
class DatabaseHelper {
  static final DatabaseHelper instance = DatabaseHelper._init();
  static Database? _database;

  DatabaseHelper._init();

  /// Veritabanı örneğini döner, veritabanı henüz açılmamışsa başlatır.
  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB(DBConstants.dbName);
    return _database!;
  }

  /// Veritabanını cihazda oluşturur/açar.
  Future<Database> _initDB(String filePath) async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, filePath);
    return await openDatabase(path, version: 1, onCreate: _createDB);
  }

  /// Veritabanı ilk oluşturulduğunda tabloyu oluşturur.
  Future<void> _createDB(Database db, int version) async {
    await db.execute('''
      CREATE TABLE ${DBConstants.tableClothes} (
        ${DBConstants.columnId} INTEGER PRIMARY KEY AUTOINCREMENT,
        ${DBConstants.columnName} TEXT NOT NULL,
        ${DBConstants.columnCategory} TEXT NOT NULL,
        ${DBConstants.columnImagePath} TEXT NOT NULL
      )
    ''');
  }

  /// Verilen ClothingItem nesnesini veritabanına ekler.
  Future<int> insertClothingItem(ClothingItem item) async {
    try {
      final db = await instance.database;
      return await db.insert(DBConstants.tableClothes, item.toMap());
    } catch (e) {
      log('insertClothingItem error: $e');
      return -1;
    }
  }

  /// Veritabanındaki tüm kıyafet kayıtlarını getirir.
  Future<List<ClothingItem>> getAllClothingItems() async {
    try {
      final db = await instance.database;
      final result = await db.query(DBConstants.tableClothes);
      return result.map((map) => ClothingItem.fromMap(map)).toList();
    } catch (e) {
      log('getAllClothingItems error: $e');
      return [];
    }
  }

  /// Verilen ID'ye sahip kıyafeti veritabanından siler.
  Future<int> deleteClothingItem(int id) async {
    try {
      final db = await instance.database;
      return await db.delete(
          DBConstants.tableClothes,
          where: '${DBConstants.columnId} = ?',
          whereArgs: [id]
      );
    } catch (e) {
      log('deleteClothingItem error: $e');
      return 0;
    }
  }

  /// Veritabanı bağlantısını kapatır.
  Future<void> close() async {
    final db = await instance.database;
    await db.close();
  }
}