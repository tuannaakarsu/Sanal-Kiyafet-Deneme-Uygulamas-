import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'features/view/main_view.dart';
import 'features/viewmodel/main_viewmodel.dart';
import 'package:my_flutter_app/features/viewmodel/settings_viewmodel.dart';
import 'package:my_flutter_app/core/services/api_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    // Ayarları yükle
    final settings = SettingsViewModel();
    await settings.loadPreferences();

    // API servisi oluştur ve verileri gönder
    final apiService = ApiService();
    await apiService.sendClothingDataToServer();

    // Provider listesi
    final providers = [
      ChangeNotifierProvider(create: (_) => MainViewModel()),
      ChangeNotifierProvider<SettingsViewModel>.value(value: settings),
    ];

    runApp(
        MultiProvider(
          providers: providers,
          child: const MyApp(),
        )
    );
  } catch (e, stackTrace) {
    debugPrint('Initialization error: $e');
    debugPrint('$stackTrace');
    // Hata durumunda bile uygulamanın açılması için çalıştır
    runApp(const MyApp());
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<SettingsViewModel>(
      builder: (context, settings, _) {
        return MaterialApp(
          debugShowCheckedModeBanner: false,
          title: 'My Flutter App',

          // Tema ayarları
          theme: ThemeData.light(useMaterial3: true),
          darkTheme: ThemeData.dark(useMaterial3: true),
          themeMode: settings.isDarkMode ? ThemeMode.dark : ThemeMode.light,

          // Giriş ekranı
          home: const MainView(),
        );
      },
    );
  }
}