import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';

class EditResults extends StatefulWidget {
  final Map<String, dynamic>? classificatoinMap;
  const EditResults({super.key, required this.classificatoinMap});

  @override
  State<EditResults> createState() => _EditResultsState();
}

class _EditResultsState extends State<EditResults> {
  List<String> dropDownMenuItems = [
    'Kana\'a',
    'Black Mesa',
    'Sosi',
    'Dogoszhi',
    'Flagstaff',
    'Tusayan',
    'Kayenta',
  ];

  late String classificationSelection;

  final Completer<GoogleMapController> _controller =
      Completer<GoogleMapController>();

  void onSaveClassificatoin(Map<String, dynamic>? classificatoinMap) async {
    final GoogleMapController controller = await _controller.future;
    LatLngBounds visibleRegion = await controller.getVisibleRegion();
    LatLng centerLatLng = LatLng(
      (visibleRegion.northeast.latitude + visibleRegion.southwest.latitude) / 2,
      (visibleRegion.northeast.longitude + visibleRegion.southwest.longitude) /
          2,
    );

    Map<String, dynamic>? newClassificationMap = classificatoinMap;

    if (classificationSelection.isNotEmpty) {
      newClassificationMap?['primaryClassification'] = classificationSelection;
    }

    newClassificationMap?['lattitude'] = centerLatLng.latitude;
    newClassificationMap?['longitude'] = centerLatLng.longitude;

    if (mounted) {
      Navigator.pop(context, newClassificationMap);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(
              height: 32,
            ),
            const SizedBox(
              width: double.infinity,
              child: FittedBox(
                fit: BoxFit.contain,
                child: Text(
                  'Edit Results',
                  style: TextStyle(
                    fontFamily: 'Uber',
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
            const SizedBox(
              height: 16,
            ),
            DropdownMenu(
              onSelected: (selectedClassification) {
                classificationSelection = selectedClassification.toString();
              },
              width: MediaQuery.of(context).size.width - 32,
              label: const Text('Classification'),
              initialSelection:
                  widget.classificatoinMap?['primaryClassification'],
              dropdownMenuEntries: dropDownMenuItems
                  .map<DropdownMenuEntry<String>>((String value) {
                return DropdownMenuEntry<String>(value: value, label: value);
              }).toList(),
            ),
            const SizedBox(
              height: 32,
            ),
            Container(
              decoration: BoxDecoration(
                  color: Theme.of(context)
                      .colorScheme
                      .secondaryContainer, // select color from current theme scheme
                  borderRadius: const BorderRadius.all(Radius.circular(5))),
              width: MediaQuery.of(context).size.width,
              height: MediaQuery.of(context).size.width,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(5),
                    child: GoogleMap(
                      mapType: MapType.terrain,
                      initialCameraPosition: CameraPosition(
                          zoom: 14,
                          target: LatLng(
                              double.parse(widget
                                  .classificatoinMap!['lattitude']
                                  .toString()),
                              double.parse(widget
                                  .classificatoinMap!['longitude']
                                  .toString()))),
                      onMapCreated: (GoogleMapController controller) {
                        _controller.complete(controller);
                      },
                    ),
                  ),
                  Container(
                    decoration: BoxDecoration(
                      color: const Color.fromARGB(140, 3, 142, 255),
                      borderRadius: BorderRadius.circular(100),
                      border: Border.all(
                        width: 2,
                        color: Colors.white,
                      ),
                    ),
                    // can we change this according to the map zoom factor?
                    height: 130,
                    width: 130,
                  ),
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.blue,
                      borderRadius: BorderRadius.circular(100),
                      border: Border.all(
                        width: 2,
                        color: Colors.white,
                      ),
                    ),
                    height: 20,
                    width: 20,
                  ),
                ],
              ),
            ),
            const SizedBox(
              height: 16,
            ),
            Center(
                child: FilledButton(
                    onPressed: () =>
                        onSaveClassificatoin(widget.classificatoinMap),
                    child: const Text('Save'))),
            Center(
                child: TextButton(
                    onPressed: () {
                      Navigator.pop(context, widget.classificatoinMap);
                    },
                    child: const Text('Back'))),
          ],
        ),
      ),
    );
  }
}
