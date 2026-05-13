import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:my_flutter_app/features/viewmodel/home_viewmodel.dart';
import 'package:my_flutter_app/features/view/settings_view.dart';
import 'package:my_flutter_app/widgets/recent_item_card.dart';
import 'package:my_flutter_app/widgets/suggestion_card.dart';

/// Ana sayfa.
/// Kullanıcıya son eklenen kıyafetleri ve önerileri gösterir.
/// Ayarlar sayfasına yönlendirme içerir.
class HomeView extends StatelessWidget {
  const HomeView({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => HomeViewModel()
        ..fetchRecentClothes()
        ..fetchSuggestions(),
      child: Consumer<HomeViewModel>(
        builder: (context, viewModel, _) {
          return Scaffold(
            appBar: AppBar(
              title: const Text('Hoş Geldin!'),
              actions: [
                IconButton(
                  icon: const Icon(Icons.settings),
                  onPressed: () {
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const SettingsView())
                    );
                  },
                ),
              ],
            ),
            body: SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Son Eklenen Kıyafetler',
                      style: TextStyle(
                        fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),

                    /// Yatay kıyafet listesi
                    SizedBox(
                        height: 170,
                        child: viewModel.recentItems.isEmpty
                            ? const Center(child: Text('Henüz kıyafet eklenmedi'),)
                            : ListView.builder(
                            scrollDirection: Axis.horizontal,
                            itemCount: viewModel.recentItems.length,
                            itemBuilder: (context, index) {
                              final item = viewModel.recentItems[index];
                              return RecentItemCard(item: item);
                            }
                        )
                    ),
                    const SizedBox(height: 16),

                    const Text(
                      'Öneriler',
                      style: TextStyle(
                          fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),

                    /// Kıyafet önerileri
                      GridView.builder(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                            crossAxisCount: 2,
                            childAspectRatio: 0.85,
                            crossAxisSpacing: 10,
                            mainAxisSpacing: 10
                        ),
                        itemCount: 4,
                        itemBuilder: (context, index) {
                          if (viewModel.suggestions.isEmpty) {
                            // Placeholder kart
                            return Container(
                              decoration: BoxDecoration(
                                color: Colors.grey.shade200,
                                borderRadius: BorderRadius.circular(12),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.grey.shade300.withAlpha((255 * 0.4).toInt()),
                                    blurRadius: 6,
                                    offset: const Offset(2, 4),
                                  )
                                ]
                              ),
                              child: const Center(
                                child: Icon(Icons.auto_awesome, color: Colors.grey, size: 40),
                              ),
                            );
                          } else {
                            return SuggestionCard(
                              suggestion: viewModel.suggestions[index],
                            );
                          }
                        }
                      ),
                  ]
              ),
            ),
          );
        },
      ),
    );
  }
}