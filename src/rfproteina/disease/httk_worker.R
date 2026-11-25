# Process-local EPA httk adapter. The only request is a bounded table of exact
# published CAS/DTXSID pairs; no model settings or study data arrive over stdin.

VERSION <- "2.7.4"
MODEL <- "3compartmentss"
SPECIES <- "Human"
MAX_ROWS <- 32L
MAX_REQUEST_BYTES <- 2048L
DRAWS <- 1000L
SEED <- 2181L
CLINT_PVALUE <- 0.05
FUP_FLOOR <- 1e-4

quiet <- function(expression) {
    # Package progress output must not become part of the worker protocol.
    before <- sink.number()
    null <- file(if (.Platform$OS.type == "windows") "NUL" else "/dev/null", "wt")
    on.exit({
        while (sink.number() > before) sink()
        close(null)
    })
    sink(null)
    suppressWarnings(suppressMessages(force(expression)))
}

read_requests <- function() {
    input <- file("stdin", "rb")
    on.exit(close(input))
    payload <- readBin(input, "raw", n = MAX_REQUEST_BYTES + 1L)
    if (!length(payload) || length(payload) > MAX_REQUEST_BYTES || any(payload == as.raw(0))) {
        stop("Invalid request size")
    }
    rows <- sub("\r$", "", strsplit(rawToChar(payload), "\n", fixed = TRUE)[[1L]])
    pattern <- "^[1-9][0-9]{1,6}-[0-9]{2}-[0-9]\tDTXSID[0-9]{7,12}$"
    if (!length(rows) || length(rows) > MAX_ROWS || !all(grepl(pattern, rows, useBytes = TRUE))) {
        stop("Invalid request rows")
    }
    requests <- strsplit(rows, "\t", fixed = TRUE)
    valid_checksum <- function(request) {
        digits <- utf8ToInt(gsub("-", "", request[[1L]], fixed = TRUE)) - 48L
        last <- length(digits)
        sum(rev(digits[-last]) * seq_len(last - 1L)) %% 10L == digits[[last]]
    }
    if (!all(vapply(requests, valid_checksum, logical(1)))) stop("Invalid CAS checksum")
    requests
}

named_columns <- function(table, expected) {
    if (!is.data.frame(table)) stop("Invalid provider table")
    column_map <- vapply(expected, function(field) {
        found <- which(tolower(names(table)) == tolower(field))
        if (length(found) != 1L) stop("Unexpected provider schema")
        found
    }, integer(1))
    result <- as.data.frame(table)[, unname(column_map), drop = FALSE]
    names(result) <- expected
    result
}

identity <- function(table, request) {
    by_cas <- which(table$CAS == request[[1L]])
    by_dtxsid <- which(table$DTXSID == request[[2L]])
    if (length(by_cas) != 1L || !identical(by_cas, by_dtxsid)) return(NULL)
    table[by_cas, , drop = FALSE]
}

numeric_parts <- function(text, counts) {
    if (length(text) != 1L || is.na(text)) return(NULL)
    text <- as.character(text)
    if (!grepl("^[^,]+(,[^,]+)*$", text)) return(NULL)
    present <- strsplit(text, ",", fixed = TRUE)[[1L]]
    values <- suppressWarnings(as.numeric(present))
    if (!(length(values) %in% counts) || any(!is.finite(values))) return(NULL)
    values
}

source_present <- function(value) {
    if (length(value) != 1L || is.na(value)) return(FALSE)
    !tolower(trimws(as.character(value))) %in% c("", "na", "n/a", "none", "unknown", "not available")
}

measured_human_inputs <- function(row) {
    if (!source_present(row$Human.Clint.Reference) || !source_present(row$Human.Funbound.plasma.Reference)) {
        return(NULL)
    }
    fup <- numeric_parts(row$Human.Funbound.plasma, c(1L, 3L))
    clint <- numeric_parts(row$Human.Clint, c(1L, 4L))
    if (is.null(fup) || is.null(clint) || fup[[1L]] < FUP_FLOOR || any(fup < 0 | fup > 1) || any(clint < 0)) {
        return(NULL)
    }
    if (length(fup) == 3L && (fup[[2L]] > fup[[1L]] || fup[[1L]] > fup[[3L]])) return(NULL)
    if (length(clint) == 4L && (clint[[2L]] > clint[[1L]] || clint[[1L]] > clint[[3L]])) return(NULL)

    # The path from the curated assay goes through EPA's normal adjustments.
    # Refuse rows that would trigger a detection-limit or significance fallback.
    # The source clearance distribution, when present, carries its own p-value.
    pvalue <- if (length(clint) == 4L) clint[[4L]] else row$Human.Clint.pValue
    if (!is.na(pvalue)) {
        pvalue <- numeric_parts(pvalue, 1L)
        if (is.null(pvalue) || pvalue[[1L]] < 0 || pvalue[[1L]] > CLINT_PVALUE) return(NULL)
    }
    # Preserve the release labels and raw central estimates, before IVIVE adjustments.
    c(sprintf("%.17g", clint[[1L]]), as.character(row$Human.Clint.Reference),
      sprintf("%.17g", fup[[1L]]), as.character(row$Human.Funbound.plasma.Reference))
}

predicted_draws <- function(cas) {
    set.seed(SEED, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
    httk::calc_mc_css(
        chem.cas = cas,
        model = MODEL,
        species = SPECIES,
        httkpop = TRUE,
        invitrouv = FALSE,
        httkpop.generate.arg.list = list(method = "direct resampling", agelim_years = c(18, 79)),
        concentration = "plasma",
        output.units = "mg/L",
        daily.dose = 1,
        samples = DRAWS,
        return.samples = TRUE,
        suppress.messages = TRUE,
        parameterize.args.list = list(
            default.to.human = FALSE,
            force.human.clint.fup = FALSE,
            adjusted.Funbound.plasma = TRUE,
            adjusted.Clint = TRUE,
            clint.pvalue.threshold = CLINT_PVALUE,
            class.exclude = TRUE,
            physchem.exclude = TRUE,
            minimum.Funbound.plasma = FUP_FLOOR
        )
    )
}

failure_row <- function(request, blocker) {
    paste(c(request, "untested", "", "", "", blocker, rep("", 4L)), collapse = "\t")
}

evaluate <- function(request, curated, covered) {
    row <- identity(curated, request)
    if (is.null(row)) return(failure_row(request, "httk_reference_identity_unavailable"))
    inputs <- measured_human_inputs(row)
    if (is.null(inputs)) return(failure_row(request, "httk_human_inputs_unavailable"))
    if (is.null(identity(covered, request))) return(failure_row(request, "httk_model_inapplicable"))

    result <- tryCatch(list(draws = predicted_draws(request[[1L]])), error = function(error) NULL)
    if (is.null(result)) return(failure_row(request, "httk_simulation_failed"))
    draws <- result$draws
    if (!is.numeric(draws) || !is.null(dim(draws)) || length(draws) != DRAWS || any(!is.finite(draws) | draws < 0)) {
        return(failure_row(request, "httk_invalid_draws"))
    }
    points <- stats::quantile(draws, probs = c(0.05, 0.50, 0.95), names = FALSE, type = 7L, na.rm = FALSE)
    if (any(!is.finite(points) | points < 0) || is.unsorted(points)) {
        return(failure_row(request, "httk_invalid_draws"))
    }
    paste(c(request, "reported", sprintf("%.17g", points), "", inputs), collapse = "\t")
}

execute <- function() {
    args <- commandArgs(trailingOnly = TRUE)
    if (length(args) != 1L || !args[[1L]] %in% c("probe", "run")) stop("Invalid command")
    if (!requireNamespace("httk", quietly = TRUE) || as.character(utils::packageVersion("httk")) != VERSION) {
        stop("Pinned EPA httk version unavailable")
    }
    if (args[[1L]] == "probe") return(paste("httk", VERSION, sep = "\t"))

    requests <- read_requests()
    invisible(Sys.setlocale("LC_NUMERIC", "C"))
    # httk deliberately looks up its published tables through the attached
    # package (a namespace-only load leaves get_cheminfo without those data).
    library("httk", character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)
    data.table::setDTthreads(1L)
    curated <- named_columns(httk::chem.physical_and_invitro.data, c(
        "CAS", "DTXSID", "Human.Clint", "Human.Clint.Reference", "Human.Clint.pValue",
        "Human.Funbound.plasma", "Human.Funbound.plasma.Reference"
    ))
    covered <- named_columns(httk::get_cheminfo(
        info = c("CAS", "DTXSID"),
        model = MODEL,
        species = SPECIES,
        default.to.human = FALSE,
        class.exclude = TRUE,
        physchem.exclude = TRUE,
        fup.ci.cutoff = TRUE,
        clint.pvalue.threshold = CLINT_PVALUE,
        suppress.messages = TRUE
    ), c("CAS", "DTXSID"))

    vapply(requests, evaluate, character(1), curated = curated, covered = covered)
}

response <- tryCatch(quiet(execute()), error = function(error) NULL)
if (is.null(response)) quit(save = "no", status = 1L, runLast = FALSE)
writeLines(response, useBytes = TRUE)
