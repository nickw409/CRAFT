import 'package:flutter/material.dart';

class LoginProvider with ChangeNotifier {
  bool _isLoggedIn = false;
  String? _fullName;
  String? _role;

  bool get isLoggedIn => _isLoggedIn;
  String? get firstName => _fullName;
  String? get role => _role;

  void login(String fullName, String role) {
    _isLoggedIn = true;
    _fullName = fullName;
    _role = role;
    notifyListeners();
  }

  void logout() {
    _isLoggedIn = false;
    _fullName = null;
    _role = null;
    notifyListeners();
  }
}
