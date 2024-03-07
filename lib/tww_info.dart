import 'package:flutter/material.dart';

class tww_info extends StatelessWidget {
  const tww_info({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Tusayan White Ware Info"),
        backgroundColor: Colors.blueAccent,
      ),
      body: const Column(
        children: [
          Text('Manufactured: Primarily between modern day Tuba City and Kayenta (Kayenta Series).\n'
              'Distribution: Throughout much of the US Southwest.\n'
              '\n'
              'Physical Characteristics:\n'
              'Paste color: Light gray to white, often with carbon streak.\n'
              'Temper: Mostly fine sand; later types may have sherd or volcanic temper.\n'
              'Surface Treatment: Polished. Slip uncommon in early types, but does occur in later types. Temper does not protrude through surface.\n'
              'Paint: Organic, soaks into surface. Blurred edges. Bidahochi type has mineral paint, sharp edges, dark.\n'
              '\n'
              'Other White Ware Types In Area:\n'
              'Cibola: Light gray to white paste, often with carbon streak; sand, sherd or both temper; no slip early, thin later; mineral paint.\n'
              'Little Colorado White Ware: Darker paste color; thick white slip; sherd temper, sometimes with sand; organic paint.\n',

            style: TextStyle(
                color: Colors.black,
                fontWeight: FontWeight.w500,
                fontSize: 16),
          ),
        ],
      ),
    );
  }
}