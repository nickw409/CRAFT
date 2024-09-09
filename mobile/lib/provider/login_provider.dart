import 'package:flutter/material.dart';

class LoginProvider with ChangeNotifier {
  bool _isLoggedIn = false;
  String? _fullName;

  bool get isLoggedIn => _isLoggedIn;
  String? get firstName => _fullName;

  void login(String fullName) {
    _isLoggedIn = true;
    _fullName = fullName;
    notifyListeners();
  }

  void logout() {
    _isLoggedIn = false;
    _fullName = null;
    notifyListeners();
  }
}
