/**
 * Convert between Uint8Array and Base64 strings
 * Allows for any encoded JS string to be converted (as opposed to atob()/btoa() which only supports latin1)
 *
 * Original implementation by madmurphy on MDN
 * @see https://developer.mozilla.org/en-US/docs/Web/API/WindowBase64/Base64_encoding_and_decoding#Solution_1_–_JavaScript%27s_UTF-16_%3E_base64
 */

function b64ToUint6(nChr) {
  return nChr > 64 && nChr < 91
    ? nChr - 65
    : nChr > 96 && nChr < 123
    ? nChr - 71
    : nChr > 47 && nChr < 58
    ? nChr + 4
    : nChr === 43
    ? 62
    : nChr === 47
    ? 63
    : 0
}

export function decodeToArray(base64string, blockSize) {
  var sB64Enc = base64string.replace(/[^A-Za-z0-9\+\/]/g, ''),
    nInLen = sB64Enc.length,
    nOutLen = blockSize
      ? Math.ceil(((nInLen * 3 + 1) >>> 2) / blockSize) * blockSize
      : (nInLen * 3 + 1) >>> 2,
    aBytes = new Uint8Array(nOutLen)

  for (
    var nMod3, nMod4, nUint24 = 0, nOutIdx = 0, nInIdx = 0;
    nInIdx < nInLen;
    nInIdx++
  ) {
    nMod4 = nInIdx & 3
    nUint24 |= b64ToUint6(sB64Enc.charCodeAt(nInIdx)) << (18 - 6 * nMod4)
    if (nMod4 === 3 || nInLen - nInIdx === 1) {
      for (nMod3 = 0; nMod3 < 3 && nOutIdx < nOutLen; nMod3++, nOutIdx++) {
        aBytes[nOutIdx] = (nUint24 >>> ((16 >>> nMod3) & 24)) & 255
      }
      nUint24 = 0
    }
  }

  return aBytes
}

function uint6ToB64(nUint6) {
  return nUint6 < 26
    ? nUint6 + 65
    : nUint6 < 52
    ? nUint6 + 71
    : nUint6 < 62
    ? nUint6 - 4
    : nUint6 === 62
    ? 43
    : nUint6 === 63
    ? 47
    : 65
}

export function encodeFromArray(bytes) {
  var eqLen = (3 - (bytes.length % 3)) % 3,
    sB64Enc = ''

  for (
    var nMod3, nLen = bytes.length, nUint24 = 0, nIdx = 0;
    nIdx < nLen;
    nIdx++
  ) {
    nMod3 = nIdx % 3
    /* Uncomment the following line in order to split the output in lines 76-character long: */
    /*
    if (nIdx > 0 && (nIdx * 4 / 3) % 76 === 0) { sB64Enc += "\r\n"; }
    */
    nUint24 |= bytes[nIdx] << ((16 >>> nMod3) & 24)
    if (nMod3 === 2 || bytes.length - nIdx === 1) {
      sB64Enc += String.fromCharCode(
        uint6ToB64((nUint24 >>> 18) & 63),
        uint6ToB64((nUint24 >>> 12) & 63),
        uint6ToB64((nUint24 >>> 6) & 63),
        uint6ToB64(nUint24 & 63)
      )
      nUint24 = 0
    }
  }

  return eqLen === 0
    ? sB64Enc
    : sB64Enc.substring(0, sB64Enc.length - eqLen) + (eqLen === 1 ? '=' : '==')
}

/**
 * URL-safe variants of Base64 conversion functions (aka base64url)
 * @see https://tools.ietf.org/html/rfc4648#section-5
 */

export function encodeFromArrayUrlSafe(bytes) {
  return encodeURIComponent(
    encodeFromArray(bytes)
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
  )
}

export function decodeToArrayUrlSafe(base64string) {
  return decodeToArray(
    decodeURIComponent(base64string)
      .replace(/-/g, '+')
      .replace(/_/g, '/')
  )
}
