/// Demo-only "lipstick" overlay built from the package's named lip contours.
///
/// This is deliberately example-app code, not library API. It exists to find
/// out what a reusable contour utility would actually need before any of it is
/// promoted into `lib/`. Everything here is derived from
/// `Face.getContour(FaceContourType.*)`; no blendshape value is consulted.
///
/// Geometry notes, all verified against real coordinates by
/// `example/integration_test/face_contours_integration_test.dart`:
///
/// * The four lip arcs pair into two closed rings. `upperLipTop` and
///   `lowerLipBottom` both run mesh index 61 -> 291 and so share those two
///   endpoints; `upperLipBottom` and `lowerLipTop` both run 78 -> 308. Joining
///   a pair therefore means appending the second arc reversed with its first
///   and last vertex dropped, giving 20 unique points per ring.
/// * Both rings come out as simple (non-self-intersecting) polygons.
/// * Their winding is NOT stable relative to each other: on a near-closed
///   mouth the inner arcs swap vertical order and the inner ring inverts. That
///   is why [PathFillType.evenOdd] is mandatory here. Under non-zero fill the
///   mouth opening would be cut out on some faces and filled solid on others.
/// * The outer ring sits slightly inside the true vermillion border, because
///   the 468-point mesh is the coarse (non-attention) variant. [dilatePixels]
///   compensates.
library;

import 'package:flutter/material.dart';
import 'package:face_detection_tflite/face_detection_tflite_native.dart';
import 'package:flutter_litert/flutter_litert.dart' show CoverFitTransform;

/// Default shade: a desaturated berry. Under [BlendMode.color] the paint's
/// saturation carries straight through, so a primary red reads as neon.
const Color kDefaultLipstickColor = Color(0xFF8C3A4A);

/// Joins two lip arcs into one closed ring (first vertex != last vertex).
///
/// [top] and [bottom] are stored in the same direction, so [bottom] is
/// appended reversed. When [sharedEndpoints] the reversed arc's first and last
/// vertices duplicate [top]'s last and first, and are dropped.
List<Point> lipRing(
  List<Point> top,
  List<Point> bottom, {
  bool sharedEndpoints = true,
}) {
  final List<Point> rev = bottom.reversed.toList();
  return <Point>[
    ...top,
    ...sharedEndpoints ? rev.sublist(1, rev.length - 1) : rev,
  ];
}

/// Shoelace signed area of a closed ring. Sign encodes winding.
double signedArea(List<Point> ring) {
  double s = 0;
  for (int i = 0; i < ring.length; i++) {
    final Point a = ring[i];
    final Point b = ring[(i + 1) % ring.length];
    s += a.x * b.y - b.x * a.y;
  }
  return s / 2;
}

/// Pushes every vertex of [ring] outward by [pixels] along its vertex normal.
///
/// The outward direction depends on the ring's winding, which is not fixed
/// (see the library doc comment), so it is derived from the sign of
/// [signedArea] rather than assumed.
List<Point> dilateRing(List<Point> ring, double pixels) {
  if (pixels == 0) return ring;
  final int n = ring.length;
  final double orient = signedArea(ring) >= 0 ? 1.0 : -1.0;

  Offset unit(Offset v) {
    final double len = v.distance;
    return len == 0 ? Offset.zero : v / len;
  }

  return <Point>[
    for (int i = 0; i < n; i++)
      () {
        final Point prev = ring[(i - 1 + n) % n];
        final Point cur = ring[i];
        final Point next = ring[(i + 1) % n];
        final Offset d1 = unit(Offset(cur.x - prev.x, cur.y - prev.y));
        final Offset d2 = unit(Offset(next.x - cur.x, next.y - cur.y));
        // Outward normal of an edge is its direction rotated by 90 degrees;
        // averaging the two adjacent edge normals gives the vertex normal.
        final Offset nrm = unit(
          Offset(d1.dy + d2.dy, -(d1.dx + d2.dx)) * orient,
        );
        return Point(cur.x + nrm.dx * pixels, cur.y + nrm.dy * pixels);
      }(),
  ];
}

/// Maps a point in source-image pixels to canvas coordinates.
///
/// Injected rather than decomposed into origin/scale because the live-camera
/// path needs a cover-fit transform with optional mirroring, which is not
/// expressible as separate x and y scales.
typedef PointMapper = Offset Function(double x, double y);

/// Appends [ring] to [path] as a closed Catmull-Rom spline through its points.
///
/// A straight polygon through 20 vertices is visibly faceted once filled, which
/// a stroked wireframe hides. [tension] 0 gives the raw polygon.
void addSmoothRing(
  Path path,
  List<Point> ring, {
  double tension = 1.0,
  required PointMapper map,
}) {
  final int n = ring.length;
  if (n < 3) return;

  Offset at(int i) {
    final Point p = ring[((i % n) + n) % n];
    return map(p.x, p.y);
  }

  path.moveTo(at(0).dx, at(0).dy);
  for (int i = 0; i < n; i++) {
    final Offset p0 = at(i - 1);
    final Offset p1 = at(i);
    final Offset p2 = at(i + 1);
    final Offset p3 = at(i + 2);
    final Offset c1 = p1 + (p2 - p0) * (tension / 6.0);
    final Offset c2 = p2 - (p3 - p1) * (tension / 6.0);
    path.cubicTo(c1.dx, c1.dy, c2.dx, c2.dy, p2.dx, p2.dy);
  }
  path.close();
}

/// Builds the fillable lip region for [face], or null when unavailable.
///
/// The mouth opening is cut out only when it is a meaningful fraction of the
/// whole mouth ([minOpenFraction]). On a closed mouth the inner ring collapses
/// and can cross itself, which would speckle an even-odd fill; the measured
/// fraction on a near-closed reference face is about 0.107.
///
/// Note this gate is purely geometric. The `jawOpen` / `mouthClose`
/// blendshapes would be the obvious alternative, but every lip landmark fed to
/// that model is unrefined coarse mesh, so they are the least reliable signal
/// available for exactly this decision.
Path? buildLipPath(
  Face face, {
  required PointMapper map,
  double dilatePixels = 0,
  double smoothing = 1.0,
  double minOpenFraction = 0.06,
}) {
  final List<Point>? upperTop = face.getContour(FaceContourType.upperLipTop);
  final List<Point>? lowerBottom =
      face.getContour(FaceContourType.lowerLipBottom);
  final List<Point>? upperBottom =
      face.getContour(FaceContourType.upperLipBottom);
  final List<Point>? lowerTop = face.getContour(FaceContourType.lowerLipTop);
  if (upperTop == null ||
      lowerBottom == null ||
      upperBottom == null ||
      lowerTop == null) {
    return null; // fast mode: no mesh, so no contours
  }

  final List<Point> outer =
      dilateRing(lipRing(upperTop, lowerBottom), dilatePixels);
  final List<Point> inner = lipRing(upperBottom, lowerTop);

  final double outerArea = signedArea(outer).abs();
  if (outerArea <= 0) return null;
  final double openFraction = signedArea(inner).abs() / outerArea;

  final Path path = Path()..fillType = PathFillType.evenOdd;
  addSmoothRing(path, outer, tension: smoothing, map: map);
  if (openFraction >= minOpenFraction) {
    addSmoothRing(path, inner, tension: smoothing, map: map);
  }
  return path;
}

/// Union of every face's lip region into one even-odd path, or null when no
/// face yields one. Lip regions never overlap between faces, so a single
/// even-odd path is safe.
Path? buildAllLipPaths(
  List<Face> faces, {
  required PointMapper map,
  double dilatePixels = 0,
  double smoothing = 1.0,
  double minOpenFraction = 0.06,
}) {
  Path? combined;
  for (final Face face in faces) {
    final Path? p = buildLipPath(
      face,
      map: map,
      dilatePixels: dilatePixels,
      smoothing: smoothing,
      minOpenFraction: minOpenFraction,
    );
    if (p == null) continue;
    combined ??= Path()..fillType = PathFillType.evenOdd;
    combined.addPath(p, Offset.zero);
  }
  return combined;
}

/// Paints lipstick over the lip region of each face.
///
/// Intended as a `foregroundPainter` on a [CustomPaint] whose child is the
/// image widget, so the fill composites against the real image pixels in the
/// same layer. That is what makes [BlendMode.color] work: it takes hue and
/// saturation from the paint and keeps luminosity from the photo, so lip
/// texture, gloss and shadow survive instead of being flattened under a decal.
class LipstickPainter extends CustomPainter {
  LipstickPainter({
    required this.faces,
    required this.originalImageSize,
    required this.color,
    this.strength = 0.85,
    this.featherPixels = 1.5,
    this.dilatePixels = 1.0,
    this.smoothing = 1.0,
    this.blendMode = BlendMode.color,
  });

  final List<Face> faces;
  final Size originalImageSize;
  final Color color;

  /// 0 = invisible, 1 = full strength.
  final double strength;

  /// Edge softness in destination pixels. The coarse mesh does not land exactly
  /// on the vermillion border, so a hard edge exposes every landmark error.
  final double featherPixels;

  /// Outward offset of the outer ring, compensating for the same.
  final double dilatePixels;

  final double smoothing;
  final BlendMode blendMode;

  @override
  void paint(Canvas canvas, Size size) {
    if (faces.isEmpty || strength <= 0) return;

    final double scaleX = size.width / originalImageSize.width;
    final double scaleY = size.height / originalImageSize.height;
    // Ring dilation is specified in source-image pixels.
    final double meanScale = (scaleX + scaleY) / 2;

    final Paint paint = Paint()
      ..style = PaintingStyle.fill
      ..blendMode = blendMode
      ..color = color.withAlpha((255 * strength.clamp(0.0, 1.0)).round());
    if (featherPixels > 0) {
      paint.maskFilter = MaskFilter.blur(
        BlurStyle.normal,
        featherPixels * meanScale,
      );
    }

    final Path? path = buildAllLipPaths(
      faces,
      map: (double x, double y) => Offset(x * scaleX, y * scaleY),
      dilatePixels: dilatePixels,
      smoothing: smoothing,
    );
    if (path != null) canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant LipstickPainter old) =>
      old.faces != faces ||
      old.color != color ||
      old.strength != strength ||
      old.featherPixels != featherPixels ||
      old.dilatePixels != dilatePixels ||
      old.smoothing != smoothing ||
      old.blendMode != blendMode ||
      old.originalImageSize != originalImageSize;
}

/// Live-camera lipstick, layered directly on top of the `CameraPreview`.
///
/// The still-image painter cannot be reused here. `CameraPreview` is a platform
/// [Texture]: its pixels never enter the Flutter canvas, so a [CustomPainter]
/// drawing with [BlendMode.color] over it has nothing to blend against and
/// produces a flat wash.
///
/// [BackdropFilter] does work, because it filters whatever is already
/// composited below it, textures included. Wrapping it in a [ClipPath] confines
/// the filter to the lip region, and a [ColorFilter] in [BlendMode.color] mode
/// applies the same hue-and-saturation-only tint the still path uses.
///
/// Known limitation: a clip has a hard edge, so there is no equivalent of the
/// still painter's `featherPixels`. Slight negative [dilatePixels] keeps the
/// boundary just inside the lip line, which hides most of it.
class LiveLipstickOverlay extends StatelessWidget {
  const LiveLipstickOverlay({
    super.key,
    required this.faces,
    required this.imageSize,
    required this.color,
    this.mirror = false,
    this.strength = 0.75,
    this.dilatePixels = -1.0,
    this.smoothing = 0.6,
  });

  final List<Face> faces;

  /// Size of the image detection ran on, post-rotation.
  final Size imageSize;

  final Color color;

  /// True for a mirrored front-camera preview.
  final bool mirror;

  final double strength;
  final double dilatePixels;
  final double smoothing;

  @override
  Widget build(BuildContext context) {
    if (faces.isEmpty || strength <= 0) return const SizedBox.shrink();
    return ClipPath(
      clipper: _LipClipper(
        faces: faces,
        imageSize: imageSize,
        mirror: mirror,
        dilatePixels: dilatePixels,
        smoothing: smoothing,
      ),
      child: BackdropFilter(
        filter: ColorFilter.mode(
          color.withAlpha((255 * strength.clamp(0.0, 1.0)).round()),
          BlendMode.color,
        ),
        child: const SizedBox.expand(),
      ),
    );
  }
}

class _LipClipper extends CustomClipper<Path> {
  const _LipClipper({
    required this.faces,
    required this.imageSize,
    required this.mirror,
    required this.dilatePixels,
    required this.smoothing,
  });

  final List<Face> faces;
  final Size imageSize;
  final bool mirror;
  final double dilatePixels;
  final double smoothing;

  @override
  Path getClip(Size size) {
    // Same cover-fit mapping CameraDetectionPainter uses, so the mask lands
    // exactly where the mesh overlay does.
    final CoverFitTransform t = CoverFitTransform.cover(
      sourceWidth: imageSize.width,
      sourceHeight: imageSize.height,
      viewWidth: size.width,
      viewHeight: size.height,
      mirror: mirror,
    );
    return buildAllLipPaths(
          faces,
          map: t.map,
          dilatePixels: dilatePixels,
          smoothing: smoothing,
        ) ??
        Path();
  }

  @override
  bool shouldReclip(_LipClipper old) =>
      old.faces != faces ||
      old.imageSize != imageSize ||
      old.mirror != mirror ||
      old.dilatePixels != dilatePixels ||
      old.smoothing != smoothing;
}
