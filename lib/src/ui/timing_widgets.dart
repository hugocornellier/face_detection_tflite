part of '../../face_detection_tflite.dart';

/// Compact tappable badge that displays the total processing time plus a
/// color-coded performance indicator (via [performanceLevel]). Tapping opens
/// a dialog with per-stage timings (detection, mesh refinement, iris
/// refinement) and a processed/skipped status per stage.
///
/// Designed as a drop-in overlay for the still-image face detection flow:
/// wrap it in a [Positioned] inside the same [Stack] as the image preview.
class TimingBadge extends StatelessWidget {
  final int totalMs;
  final int? detectionMs;
  final int? meshMs;
  final int? irisMs;
  final bool meshEnabled;
  final bool irisEnabled;
  final bool meshProcessed;
  final bool irisProcessed;

  const TimingBadge({
    super.key,
    required this.totalMs,
    this.detectionMs,
    this.meshMs,
    this.irisMs,
    this.meshEnabled = false,
    this.irisEnabled = false,
    this.meshProcessed = false,
    this.irisProcessed = false,
  });

  @override
  Widget build(BuildContext context) {
    final perf = performanceLevel(totalMs);
    return GestureDetector(
      onTap: () => _showDetails(context),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black.withAlpha(179),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(perf.icon, size: 14, color: perf.color),
            const SizedBox(width: 6),
            Text(
              '${totalMs}ms',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 12,
              ),
            ),
            const SizedBox(width: 4),
            Text(
              perf.label,
              style: TextStyle(color: perf.color, fontSize: 12),
            ),
            const SizedBox(width: 4),
            const Icon(Icons.info_outline, size: 12, color: Colors.white54),
          ],
        ),
      ),
    );
  }

  void _showDetails(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: const [
            Icon(Icons.timer, color: Colors.blue),
            SizedBox(width: 8),
            Text('Processing Details'),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const _StatusRow(
                label: 'Detection', processed: true, color: Colors.green),
            _StatusRow(
              label: 'Mesh',
              processed: meshProcessed,
              color: meshEnabled ? Colors.green : Colors.grey,
            ),
            _StatusRow(
              label: 'Iris',
              processed: irisProcessed,
              color: irisEnabled ? Colors.green : Colors.grey,
            ),
            const Divider(height: 16),
            if (detectionMs != null)
              _TimingRow(
                  label: 'Detection', ms: detectionMs!, color: Colors.green),
            if (meshMs != null)
              _TimingRow(
                  label: 'Mesh Refinement', ms: meshMs!, color: Colors.pink),
            if (irisMs != null)
              _TimingRow(
                  label: 'Iris Refinement',
                  ms: irisMs!,
                  color: Colors.blueAccent),
            _TimingRow(
                label: 'Total', ms: totalMs, color: Colors.blue, isBold: true),
            const SizedBox(height: 12),
            _PerformanceIndicator(totalMs: totalMs),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
}

class _TimingRow extends StatelessWidget {
  final String label;
  final int ms;
  final Color color;
  final bool isBold;

  const _TimingRow({
    required this.label,
    required this.ms,
    required this.color,
    this.isBold = false,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(
                  color: color,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                label,
                style: TextStyle(
                  fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
                  fontSize: isBold ? 15 : 14,
                ),
              ),
            ],
          ),
          Text(
            '${ms}ms',
            style: TextStyle(
              fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
              fontSize: isBold ? 15 : 14,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}

class _StatusRow extends StatelessWidget {
  final String label;
  final bool processed;
  final Color color;

  const _StatusRow({
    required this.label,
    required this.processed,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Icon(
            processed ? Icons.check_circle : Icons.pending,
            size: 16,
            color: color,
          ),
          const SizedBox(width: 8),
          Text(label, style: const TextStyle(fontSize: 14)),
          const Spacer(),
          Text(
            processed ? 'Processed' : 'Skipped',
            style: TextStyle(
              fontSize: 12,
              color: color,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}

class _PerformanceIndicator extends StatelessWidget {
  final int totalMs;

  const _PerformanceIndicator({required this.totalMs});

  @override
  Widget build(BuildContext context) {
    final perf = performanceLevel(totalMs);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: perf.color.withAlpha(26),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: perf.color.withAlpha(77)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(perf.icon, size: 16, color: perf.color),
          const SizedBox(width: 6),
          Text(
            perf.label,
            style: TextStyle(
              color: perf.color,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }
}
