"""Binary file extensions to skip for text-based operations.

These files can't be meaningfully compared as text and are often large.
Ported from free-code src/constants/files.ts.
"""

BINARY_EXTENSIONS = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".tif",
    # Videos
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpeg", ".mpg",
    # Audio
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".opus",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz", ".z", ".tgz", ".iso",
    # Executables/binaries
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".obj", ".lib",
    ".app", ".msi", ".deb", ".rpm",
    # Documents (exclude .pdf — text-based, agents may want to inspect)
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Bytecode / VM artifacts
    ".pyc", ".pyo", ".class", ".jar", ".war", ".ear", ".node", ".wasm", ".rlib",
    # Database files
    ".sqlite", ".sqlite3", ".db", ".mdb", ".idx",
    # Design / 3D
    ".psd", ".ai", ".eps", ".sketch", ".fig", ".xd", ".blend", ".3ds", ".max",
    # Flash
    ".swf", ".fla",
    # Lock/profiling data
    ".lockb", ".dat", ".data",
})

# Container document formats (OOXML/ODF/EPUB zips, OLE compound, RTF) that a
# plain-text write can NEVER produce validly. read_file auto-extracts these to
# text, so a model that "read" report.docx and writes the text back via
# write_file/patch silently destroys the document. PDF is deliberately absent:
# raw PDF syntax is text-authorable, so only overwrites are dangerous (handled
# by the write guard via is_pdf_path).
OPAQUE_DOCUMENT_EXTENSIONS = frozenset({
    ".doc", ".docx", ".docm",
    ".xls", ".xlsx", ".xlsm", ".xlsb",
    ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".odt", ".ods", ".odp",
    ".rtf", ".epub",
})


def _has_extension_in(path: str, extensions: frozenset) -> bool:
    """Pure string check on the final ``.suffix`` (case-insensitive), no I/O."""
    dot = path.rfind(".")
    return dot != -1 and path[dot:].lower() in extensions


def has_binary_extension(path: str) -> bool:
    """True when the path has a binary extension. Pure string check, no I/O."""
    return _has_extension_in(path, BINARY_EXTENSIONS)


def has_opaque_document_extension(path: str) -> bool:
    """True when the path names an opaque container document (.docx etc.)."""
    return _has_extension_in(path, OPAQUE_DOCUMENT_EXTENSIONS)


def is_pdf_path(path: str) -> bool:
    """True when the path has a .pdf extension. Pure string check, no I/O."""
    return path.lower().endswith(".pdf")
