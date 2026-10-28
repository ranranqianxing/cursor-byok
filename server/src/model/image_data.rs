//! Decodes image payloads carried by Cursor and OpenAI-compatible messages.
use base64::{engine::general_purpose::STANDARD, Engine};

use crate::{Error, Result};

pub(crate) fn is_image_mime(mime: &str) -> bool {
    let mime = mime.trim();
    mime.starts_with("image/") && mime.len() > "image/".len() && !mime.contains(';')
}

pub(crate) fn sniff_mime(data: &[u8]) -> Option<&'static str> {
    if data.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("image/png")
    } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// Decodes a data URL or raw standard-base64 image into bytes and a MIME type.
pub(crate) fn image_from_encoded(
    payload: &str,
    mime_hint: Option<&str>,
) -> Result<(String, Vec<u8>)> {
    let payload = payload.trim();
    if payload.is_empty() {
        return Err(Error::Protocol("image payload is empty".into()));
    }
    if payload.starts_with("http://") || payload.starts_with("https://") {
        return Err(Error::Protocol(
            "remote image URLs are not inlined; send base64 image bytes".into(),
        ));
    }
    if let Some(rest) = payload.strip_prefix("data:") {
        let (meta, encoded) = rest
            .split_once(',')
            .ok_or_else(|| Error::Protocol("invalid image data URL".into()))?;
        if !meta
            .split(';')
            .any(|part| part.trim().eq_ignore_ascii_case("base64"))
        {
            return Err(Error::Protocol("image data URL is not base64".into()));
        }
        let mime = meta.split(';').next().unwrap_or("").trim();
        if !is_image_mime(mime) {
            return Err(Error::Protocol(format!(
                "image data URL has invalid MIME type: {mime}"
            )));
        }
        let data = decode_base64(encoded)?;
        if data.is_empty() {
            return Err(Error::Protocol("image data is empty".into()));
        }
        return Ok((mime.to_string(), data));
    }
    let data = decode_base64(payload)?;
    if data.is_empty() {
        return Err(Error::Protocol("image data is empty".into()));
    }
    let mime = mime_hint
        .map(str::trim)
        .filter(|mime| is_image_mime(mime))
        .map(str::to_string)
        .or_else(|| sniff_mime(&data).map(str::to_string))
        .ok_or_else(|| Error::Protocol("image is missing a MIME type".into()))?;
    Ok((mime, data))
}

/// Normalizes inline `SelectedImage` bytes.
///
/// Current clients send raw image bytes. Some also send a data URL or standard
/// base64 text in that field; both have to become the raw bytes the provider
/// data URL is built from, or the image is wrapped a second time and the model
/// cannot read it.
pub(crate) fn materialize_inline(data: Vec<u8>, mime_hint: &str) -> Result<(String, Vec<u8>)> {
    if let Some(sniffed) = sniff_mime(&data) {
        let mime_type = if is_image_mime(mime_hint) {
            mime_hint.to_string()
        } else {
            sniffed.to_string()
        };
        return Ok((mime_type, data));
    }
    if let Ok(text) = std::str::from_utf8(&data) {
        let text = text.trim();
        if text.starts_with("data:") || !is_image_mime(mime_hint) {
            if let Ok(image) =
                image_from_encoded(text, is_image_mime(mime_hint).then_some(mime_hint))
            {
                return Ok(image);
            }
        }
    }
    if is_image_mime(mime_hint) && !data.is_empty() {
        return Ok((mime_hint.to_string(), data));
    }
    Err(Error::Protocol(format!(
        "selected image has invalid MIME type: {mime_hint}"
    )))
}

fn decode_base64(payload: &str) -> Result<Vec<u8>> {
    let compact: String = payload
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    STANDARD
        .decode(compact)
        .map_err(|error| Error::Protocol(format!("invalid image base64: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_url_and_raw_base64_decode_to_the_same_bytes() {
        let from_url = image_from_encoded("data:image/png;base64,AQID", None).unwrap();
        let from_raw = image_from_encoded("AQID", Some("image/png")).unwrap();
        assert_eq!(from_url, ("image/png".into(), vec![1, 2, 3]));
        assert_eq!(from_raw, from_url);
    }

    #[test]
    fn textual_data_url_bytes_are_not_wrapped_again() {
        let text = b"data:image/png;base64,AQID".to_vec();
        let (mime, data) = materialize_inline(text, "").unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn raw_png_bytes_keep_their_mime() {
        let mut data = vec![0x89, b'P', b'N', b'G'];
        data.extend_from_slice(&[1, 2, 3, 4]);
        let (mime, bytes) = materialize_inline(data.clone(), "").unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(bytes, data);
    }
}
