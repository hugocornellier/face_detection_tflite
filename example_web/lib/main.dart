// ignore_for_file: public_member_api_docs

import 'package:flutter/material.dart';
import 'package:flutter_litert/flutter_litert.dart'
    show configureLiteRtWebLoader;

import 'live_camera_screen.dart';
import 'still_image_screen.dart';
import 'video_file_screen.dart';

void main() {
  // Pass the LiteRT.js wasm location as a DIRECTORY (trailing slash) so its
  // runtime feature probe picks the right build per engine: the fast
  // relaxed-SIMD build on Chrome/Firefox, the compat build on Safari (which
  // ships relaxed SIMD behind a default-off flag and rejects the fast build
  // at parse time). A full .js file path would bypass that selection.
  configureLiteRtWebLoader(
    wasmUrl: 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.4.0/wasm/',
  );
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Detection Web',
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    // Deep-link hook (also used by automated validation):
    // /?screen=camera|video|still opens a demo screen directly.
    final String? screen = Uri.base.queryParameters['screen'];
    if (screen == null) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final Widget? target = switch (screen) {
        'camera' => const LiveCameraScreen(),
        'video' => const VideoFileScreen(),
        'still' => const StillImageScreen(),
        _ => null,
      };
      if (target != null) {
        Navigator.of(context).push(
          MaterialPageRoute<void>(builder: (_) => target),
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Face Detection - Web')),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 720),
            child: GridView.count(
              crossAxisCount: 2,
              shrinkWrap: true,
              mainAxisSpacing: 16,
              crossAxisSpacing: 16,
              children: [
                _DemoCard(
                  icon: Icons.videocam,
                  title: 'Live Camera',
                  subtitle: 'Real-time webcam detection via getUserMedia.',
                  onTap: () => Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => const LiveCameraScreen(),
                    ),
                  ),
                ),
                _DemoCard(
                  icon: Icons.image,
                  title: 'Still Image',
                  subtitle: 'Upload an image and detect faces, mesh, irises.',
                  onTap: () => Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => const StillImageScreen(),
                    ),
                  ),
                ),
                _DemoCard(
                  icon: Icons.movie,
                  title: 'Video File',
                  subtitle: 'Upload a video and detect faces frame by frame '
                      'with a live overlay.',
                  onTap: () => Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => const VideoFileScreen(),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _DemoCard extends StatelessWidget {
  const _DemoCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 40),
              const SizedBox(height: 12),
              Text(
                title,
                style: Theme.of(context).textTheme.titleLarge,
              ),
              const SizedBox(height: 6),
              Text(
                subtitle,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
