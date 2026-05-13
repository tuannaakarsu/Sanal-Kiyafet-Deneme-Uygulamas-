import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:my_flutter_app/features/viewmodel/closet_viewmodel.dart';
import 'package:my_flutter_app/features/view/add_clothing_view.dart';
import 'package:my_flutter_app/widgets/clothing_card.dart';

/// Dolap sayfası. Kullanıcıya kıyafetleri gösterir, arama & filtreleme yapar.
/// Ayrıca isSelectionMode aktifse kullanıcı kıyafet seçebilir.
class ClosetView extends StatelessWidget {
  final bool isSelectionMode;

  const ClosetView({super.key, this.isSelectionMode = false});

  /// Silme işlemi için onay kutusu gösterir ve onaylanırsa siler.
  void _confirmDelete(BuildContext context, int id, ClosetViewModel viewModel) {
    showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Silmek istediğinize emin misiniz?'),
          content: const Text('Bu kıyafet dolabınızdan kalıcı olarak silinecek.'),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('İptal'),
            ),
            TextButton(
                onPressed: () {
                  Navigator.pop(context);
                  viewModel.deleteClothing(id);
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Kıyafet başarıyla silindi.')),
                  );
                },
                child: const Text('Sil', style: TextStyle(color: Colors.red),)
            )
          ],
        )
    );
  }

  /// Kategori ve arama filtrelerini içeren üst bar.
  Widget _buildFilterBar(ClosetViewModel viewModel) {
    final categories = ['Tümü', 'Tişört', 'Pantolon', 'Ceket', 'Etek', 'Elbise'];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        TextField(
          decoration: const InputDecoration(
            hintText: 'Kıyafet adına göre ara...',
            prefixIcon: Icon(Icons.search),
            border: OutlineInputBorder(),
          ),
          onChanged: viewModel.setSearchQuery,
        ),
        const SizedBox(height: 10),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(
            children: categories.map((category) {
              final isSelected = viewModel.selectedCategory == category ||
                  (category == 'Tümü' && viewModel.selectedCategory == null);

              return Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                  child: ChoiceChip(
                      label: Text(category),
                      selected: isSelected,
                      onSelected: (_) => viewModel.setCategory(
                        category == 'Tümü' ? null : category,
                      ),
                  ),
              );
            }).toList(),
          ),
        )
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => ClosetViewModel()..fetchClothes(),
      child: Consumer<ClosetViewModel>(
        builder: (context, viewModel, _) {
          return Scaffold(
            appBar: AppBar(
              title: const Text('Dolabım'),
              centerTitle: true,
              actions: [
                IconButton(
                  iconSize: 40,
                  padding: const EdgeInsets.only(right: 20, bottom: 5),
                  icon: const Icon(Icons.add),
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const AddClothingView()),
                    ).then((_) => viewModel.fetchClothes()); // geri dönünce listeyi yenile
                  },
                ),
              ],
            ),
            body: Padding(
              padding: const EdgeInsets.all(12.0),
              child: Column(
                children: [
                  _buildFilterBar(viewModel),
                  const SizedBox(height: 12,),
                  Expanded(
                      child: viewModel.filteredClothes.isEmpty
                          ? const Center(child: Text('Eşleşen kıyafet yok'))
                          : GridView.builder(
                          itemCount: viewModel.filteredClothes.length,
                          gridDelegate:
                            const SliverGridDelegateWithFixedCrossAxisCount(
                                crossAxisCount: 2,
                                crossAxisSpacing: 10,
                                mainAxisSpacing: 10,
                                childAspectRatio: 0.75,
                            ),
                          itemBuilder: (context, index) {
                            final item = viewModel.filteredClothes[index];
                            return GestureDetector(
                              onTap: () {
                                if (isSelectionMode) {
                                  Navigator.pop(context, item.imagePath); //Seçilen kıyafeti geri döndür
                                }
                              },
                              child: ClothingCard(
                                  item: item,
                                  onDelete: isSelectionMode ? null
                                      : () => _confirmDelete(context, item.id!, viewModel)
                              ),
                            );
                          }
                      )
                  )
                ],
              )
              ),
            );
        },
      ),
    );
  }
  }
