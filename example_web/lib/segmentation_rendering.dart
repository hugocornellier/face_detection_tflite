// ignore_for_file: public_member_api_docs

import 'dart:js_interop';

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

/// Class labels and overlay colors for the multiclass segmentation model.
/// These match the native example's `kSegmentationClassLabels` and
/// `kSegmentationClassColors`, which the package does not export on web.
const List<String> kClassLabels = [
  'BG',
  'Hair',
  'Body',
  'Face',
  'Clothes',
  'Other',
];

const List<Color> kClassColors = [
  Color(0x99A0A0A0),
  Color(0x99CD853F),
  Color(0x88FFA500),
  Color(0x88FF69B4),
  Color(0x9900BFFF),
  Color(0x9940E0D0),
];

/// The mask region holding real data, excluding letterbox padding.
({int x0, int y0, int x1, int y1}) maskValidRegion(SegmentationMask mask) {
  return (
    x0: (mask.padding[2] * mask.width).round(),
    y0: (mask.padding[0] * mask.height).round(),
    x1: ((1.0 - mask.padding[3]) * mask.width).round(),
    y1: ((1.0 - mask.padding[1]) * mask.height).round(),
  );
}

/// Draws [mask] over the frame already rendered on [ctx], compositing with
/// alpha like the native overlay painters.
///
/// The mask is rasterized at mask resolution into [scratch] (a reusable
/// offscreen canvas), then scaled onto the destination via `drawImage`, which
/// both alpha-composites and lets the browser do the upsampling. Writing
/// `putImageData` directly to the display canvas would instead *replace* the
/// frame pixels, leaving black outside the mask.
///
/// [threshold] < 0 means soft alpha (proportional to probability); otherwise
/// pixels at or above the threshold get the full mask color alpha.
/// [showAllClasses] renders a per-pixel argmax in [kClassColors] with class
/// labels at centroids; [classIndex] isolates a single multiclass channel.
/// [mirrored] must be true when the display canvas is CSS-flipped (front
/// camera) so the class labels stay readable.
void drawSegmentationOverlay({
  required web.CanvasRenderingContext2D ctx,
  required web.HTMLCanvasElement scratch,
  required SegmentationMask mask,
  required int destWidth,
  required int destHeight,
  required Color maskColor,
  required double threshold,
  int? classIndex,
  bool showAllClasses = false,
  bool mirrored = false,
}) {
  final v = maskValidRegion(mask);
  final vw = v.x1 - v.x0;
  final vh = v.y1 - v.y0;
  if (vw <= 0 || vh <= 0) return;

  final sctx = scratch.getContext('2d') as web.CanvasRenderingContext2D;
  if (scratch.width != vw) scratch.width = vw;
  if (scratch.height != vh) scratch.height = vh;
  final imageData = sctx.createImageData(vw.toJS, vh);
  final rgba = imageData.data.toDart;

  List<int>? counts;
  List<double>? sumX;
  List<double>? sumY;

  if (showAllClasses && mask is MulticlassSegmentationMask) {
    final classData = mask.internalClassData;
    final minProb = threshold < 0 ? 0.5 : threshold;
    counts = List<int>.filled(6, 0);
    sumX = List<double>.filled(6, 0);
    sumY = List<double>.filled(6, 0);
    for (int y = v.y0; y < v.y1; y++) {
      for (int x = v.x0; x < v.x1; x++) {
        final idx = y * mask.width + x;
        int winner = 0;
        double maxProb = classData[idx * 6];
        for (int c = 1; c < 6; c++) {
          final p = classData[idx * 6 + c];
          if (p > maxProb) {
            maxProb = p;
            winner = c;
          }
        }
        if (maxProb < minProb) continue;
        final color = kClassColors[winner];
        final off = ((y - v.y0) * vw + (x - v.x0)) * 4;
        rgba[off] = (color.r * 255).round();
        rgba[off + 1] = (color.g * 255).round();
        rgba[off + 2] = (color.b * 255).round();
        rgba[off + 3] = (maxProb * color.a * 255).round();
        counts[winner]++;
        sumX[winner] += x - v.x0;
        sumY[winner] += y - v.y0;
      }
    }
  } else {
    final probs = classIndex != null && mask is MulticlassSegmentationMask
        ? mask.classMask(classIndex)
        : mask.internalData;
    final r = (maskColor.r * 255).round();
    final g = (maskColor.g * 255).round();
    final b = (maskColor.b * 255).round();
    final baseAlpha = maskColor.a * 255;
    for (int y = v.y0; y < v.y1; y++) {
      for (int x = v.x0; x < v.x1; x++) {
        final p = probs[y * mask.width + x].clamp(0.0, 1.0);
        final a = threshold < 0 ? p : (p >= threshold ? 1.0 : 0.0);
        if (a == 0.0) continue;
        final off = ((y - v.y0) * vw + (x - v.x0)) * 4;
        rgba[off] = r;
        rgba[off + 1] = g;
        rgba[off + 2] = b;
        rgba[off + 3] = (a * baseAlpha).round();
      }
    }
  }

  sctx.putImageData(imageData, 0, 0);
  ctx.drawImage(scratch, 0, 0, destWidth, destHeight);

  if (counts != null) {
    _drawClassLabels(
      ctx,
      counts,
      sumX!,
      sumY!,
      vw,
      vh,
      destWidth,
      destHeight,
      mirrored,
    );
  }
}

void _drawClassLabels(
  web.CanvasRenderingContext2D ctx,
  List<int> counts,
  List<double> sumX,
  List<double> sumY,
  int maskW,
  int maskH,
  int destWidth,
  int destHeight,
  bool mirrored,
) {
  final scaleX = destWidth / maskW;
  final scaleY = destHeight / maskH;
  ctx.save();
  if (mirrored) {
    // The display canvas is CSS-flipped, so pre-flip the text here to keep it
    // readable while still landing on the class centroid.
    ctx.translate(destWidth, 0);
    ctx.scale(-1, 1);
  }
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#ffffff'.toJS;
  ctx.shadowColor = '#000000';
  ctx.shadowBlur = 3;
  for (int c = 0; c < 6; c++) {
    if (counts[c] <= 100) continue;
    final cx = sumX[c] / counts[c] * scaleX;
    final cy = sumY[c] / counts[c] * scaleY;
    final drawX = mirrored ? destWidth - cx : cx;
    ctx.fillText(kClassLabels[c], drawX, cy);
  }
  ctx.restore();
}

/// Composites the person cut out of [frame] over a solid background color,
/// mirroring the native virtual background overlay. [scratch] receives the
/// mask alpha at mask resolution; [composite] receives the person cutout at
/// destination resolution.
void drawVirtualBackground({
  required web.CanvasRenderingContext2D ctx,
  required web.HTMLCanvasElement scratch,
  required web.HTMLCanvasElement composite,
  required web.CanvasImageSource frame,
  required SegmentationMask mask,
  required int destWidth,
  required int destHeight,
}) {
  final v = maskValidRegion(mask);
  final vw = v.x1 - v.x0;
  final vh = v.y1 - v.y0;
  if (vw <= 0 || vh <= 0) return;

  final sctx = scratch.getContext('2d') as web.CanvasRenderingContext2D;
  if (scratch.width != vw) scratch.width = vw;
  if (scratch.height != vh) scratch.height = vh;
  final imageData = sctx.createImageData(vw.toJS, vh);
  final rgba = imageData.data.toDart;
  final probs = mask.internalData;
  for (int y = v.y0; y < v.y1; y++) {
    for (int x = v.x0; x < v.x1; x++) {
      final p = probs[y * mask.width + x].clamp(0.0, 1.0);
      rgba[((y - v.y0) * vw + (x - v.x0)) * 4 + 3] = (p * 255).round();
    }
  }
  sctx.putImageData(imageData, 0, 0);

  if (composite.width != destWidth) composite.width = destWidth;
  if (composite.height != destHeight) composite.height = destHeight;
  final cctx = composite.getContext('2d') as web.CanvasRenderingContext2D;
  cctx.globalCompositeOperation = 'source-over';
  cctx.clearRect(0, 0, destWidth, destHeight);
  cctx.drawImage(frame, 0, 0, destWidth, destHeight);
  cctx.globalCompositeOperation = 'destination-in';
  cctx.drawImage(scratch, 0, 0, destWidth, destHeight);
  cctx.globalCompositeOperation = 'source-over';

  ctx.fillStyle = 'rgb(20,20,80)'.toJS;
  ctx.fillRect(0, 0, destWidth, destHeight);
  ctx.drawImage(composite, 0, 0);
}
