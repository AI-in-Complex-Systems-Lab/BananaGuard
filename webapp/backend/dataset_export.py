import io
import zipfile


EXPORTABLE_STATUSES = {"approved", "corrected"}


def build_yolo_export(
    records,
    frame_width,
    frame_height,
    frame_images,
):
    """Build a YOLO-format dataset zip from reviewed detections.

    records: review detection records as stored by ReviewStore.
    frame_width, frame_height: source video dimensions in pixels.
    frame_images: dict of {frame_number: jpg_bytes} for the frames
    that were successfully extracted from the source video.

    Returns (zip_bytes, included_frame_count, included_detection_count),
    or (None, 0, 0) if there is nothing exportable.
    """
    qualifying = [
        record
        for record in records
        if record["status"] in EXPORTABLE_STATUSES
    ]

    by_frame = {}

    for record in qualifying:
        by_frame.setdefault(record["frame"], []).append(
            record
        )

    exportable_frames = [
        (frame_number, frame_records)
        for frame_number, frame_records in sorted(
            by_frame.items()
        )
        if frame_images.get(frame_number) is not None
    ]

    if not exportable_frames:
        return None, 0, 0

    classes = sorted(
        {record["label"] for record in qualifying}
    )
    class_index = {
        label: index for index, label in enumerate(classes)
    }

    buffer = io.BytesIO()
    included_detections = 0

    with zipfile.ZipFile(
        buffer, "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for (
            frame_number,
            frame_records,
        ) in exportable_frames:
            image_bytes = frame_images[frame_number]
            label_lines = []

            for record in frame_records:
                x, y, width, height = record["box"]

                x_center = (
                    x + width / 2
                ) / frame_width
                y_center = (
                    y + height / 2
                ) / frame_height
                norm_width = width / frame_width
                norm_height = height / frame_height

                class_id = class_index[record["label"]]

                label_lines.append(
                    f"{class_id} {x_center:.6f} "
                    f"{y_center:.6f} {norm_width:.6f} "
                    f"{norm_height:.6f}"
                )

            archive.writestr(
                f"images/frame_{frame_number}.jpg",
                image_bytes,
            )

            archive.writestr(
                f"labels/frame_{frame_number}.txt",
                "\n".join(label_lines) + "\n",
            )

            included_detections += len(frame_records)

        archive.writestr(
            "classes.txt", "\n".join(classes) + "\n"
        )

        data_yaml = (
            "path: .\n"
            "train: images\n"
            "val: images\n"
            f"nc: {len(classes)}\n"
            f"names: {classes}\n"
        )

        archive.writestr("data.yaml", data_yaml)

    return (
        buffer.getvalue(),
        len(exportable_frames),
        included_detections,
    )
