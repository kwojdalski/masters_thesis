\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pracamgr_wne}[2006/05/05 v0.5.0 Praca magisterska]


\def\@baseclass{report}
\def\@baseclass{article} % fix for qmd
\@ifundefined{l@chapter}{\newcounter{chapter}%
  \newcommand*\l@chapter{\@dottedtocline{0}{0em}{1.5em}}}{}
% Annex D's English edition labels this line "Licencjat/Magister
% (bachelor/master) degree thesis", so the degree word is English here rather
% than the Polish "magisterska"/"licencjacka" the option names still use.
\def\@rodzajpracy{Magister (master)}
\DeclareOption{licencjacka}{\def\@rodzajpracy{Licencjat (bachelor)}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\@baseclass}}
\PassOptionsToClass{a4paper,twoside,openright,12pt}{\@baseclass}
\ProcessOptions

\LoadClass{\@baseclass}

% List of Appendices, structured like \listoffigures/\listoftables below.
% Entries are registered per-appendix via \addcontentsline{lap}{section}{...}
% in 99-appendix.qmd. Exposed as its own \listofappendices command (rather
% than a Quarto front-matter option, since Pandoc's PDF template has no
% native "lap" switch) so masters-thesis.qmd can call it explicitly wherever
% it belongs in the back matter.
%
% Extension is "lap" (not "loa"): the algorithm package (loaded in
% masters-thesis.qmd's header-includes for the pseudocode-block algorithms)
% already writes its own List-of-Algorithms entries to a .loa file via every
% \caption{...} inside an algorithm environment. Reusing "loa" here silently
% merged those algorithm captions into this list, run together with no line
% breaks or page numbers since they weren't logged at the "section" level
% tocloft formats.
\usepackage{tocloft}
\newlistof{appendices}{lap}{List of Appendices}

% tocloft's \newlistof hard-codes \Huge\bfseries for a new list's title font,
% even in article mode (no chapters) where tocloft's own \listoffigures and
% \listoftables titles are set to \Large\bfseries. Left as-is the "List of
% Appendices" heading prints two size steps larger than "List of Figures" and
% "List of Tables". Match it to \cftloftitlefont so all three lists -- and the
% table of contents, also \Large -- share one heading size, as WNE UW rule 15
% requires (the list of lists is treated like the table of contents).
\renewcommand{\cftlaptitlefont}{\cftloftitlefont}

% Annex D p. 15 sets the whole list-of-lists block at the body's 12 pt: its
% "LIST OF APPENDICES" heading and its "List of shorts" / "List of tables" /
% "List of pictures" sub-headings all measure 12.0 pt bold at the 25 mm left
% margin, consistent with "Font of thesis: Times New Roman, size 12 points.
% The same applies to the font of chapters and subsections" (p. 5). These
% three titles printed at 17.2 pt, tocloft's \Large default, which also made
% them inconsistent with BIBLIOGRAPHY and the glossary heading in this same
% document at 12 pt.
% Annex D heads these blocks in capitals ("TABLE OF CONTENTS" p. 4, "LIST OF
% APPENDICES" p. 15). Not applied: Pandoc's template re-sets \contentsname,
% \listtablename and \listfigurename after this class loads, so renewing them
% here has no effect, and capitalising only the appendices list -- the one
% title this class owns outright, via \newlistof above -- would leave the four
% headings inconsistent with each other. Left in title case as a set.
\renewcommand{\cfttoctitlefont}{\normalsize\bfseries}
\renewcommand{\cftlottitlefont}{\normalsize\bfseries}
\renewcommand{\cftloftitlefont}{\normalsize\bfseries}

% Annex D prints its entries as "Tabel 1. Zestawienie wartosci ... 23" and
% "Pic. 1. Schemat ... 12" -- the label word is part of the entry. WNE UW
% rule 15 assumes it is there, asking that wrapped lines be indented so
% "Table 12." or "Fig. 13." stays visible. These lists printed a bare "1.",
% so there was no label to keep visible. The words match the body captions
% ("Table N." / "Figure N."), so the two agree.
% tocloft's \cfttabpresnum / \cftfigpresnum cannot be used for this: line 379
% below redefines \numberline globally to typeset "#1." in a fixed box, which
% is what puts the period after each number throughout the contents, and it
% bypasses tocloft's prefix mechanism entirely. The label is therefore added
% by wrapping \l@table and \l@figure, which receive the already-formatted
% entry and can prepend to it.
\let\@oldlottable\l@table
\renewcommand*\l@table[2]{\@oldlottable{Table~#1}{#2}}
\let\@oldlotfigure\l@figure
\renewcommand*\l@figure[2]{\@oldlotfigure{Fig.~#1}{#2}}
% Room for the widest label plus its number, or tocloft truncates it.
\settowidth{\cfttabnumwidth}{Table 88.\quad}
\settowidth{\cftfignumwidth}{Fig. 88.\quad}
% Annex D starts its entries at the 25 mm left margin, not indented.
\setlength{\cfttabindent}{0pt}
\setlength{\cftfigindent}{0pt}
\setlength{\cftappendicesindent}{0pt}

% tocloft's own \listoffigures/\listoftables definitions are deferred via
% \AtBeginDocument (see tocloft.sty), which runs after any ordinary preamble
% \renewcommand -- so redefining these commands here must ALSO be deferred
% via \AtBeginDocument, registered after \usepackage{tocloft} above, so this
% hook fires after tocloft's own and therefore wins.
%
% WNE UW rule 15 specifies the back-matter "zestawienie spisow" order as
% abbreviations, tables, figures, appendices-list, appendices-content. Each
% of the three lists below starts with a \clearpage so masters-thesis.qmd can
% call \listoftables, \listoffigures, \listofappendices back-to-back in that
% order and have each begin on its own page. \clearpage, not
% \cleardoublepage: the document is twoside, so \cleardoublepage inserts a
% blank verso to force a recto opening, which put an empty page before each
% list. The rest of the document breaks sections with \newpage / \pagebreak
% and never forces recto openings, and rule 15 asks for the list of lists to
% be handled like the table of contents, which is not forced to a recto
% either -- so a plain page break is what matches.
\AtBeginDocument{%
  \let\oldlistoffigures\listoffigures
  \renewcommand{\listoffigures}{%
    \clearpage
    \oldlistoffigures
  }
  \let\oldlistoftables\listoftables
  \renewcommand{\listoftables}{%
    \clearpage
    \oldlistoftables
  }
  \let\oldlistofappendices\listofappendices
  \renewcommand{\listofappendices}{%
    \clearpage
    \oldlistofappendices
    \clearpage
  }
}
% List-of-Appendices entries carry the appendix letter (A, B, ...), matching the
% appendix headings and the table of contents. Each appendix section in
% 99-appendix.qmd writes its own entry as
% \addcontentsline{lap}{appendices}{\protect\numberline{\thesection}Title};
% widen the label box from tocloft's default so "A." .. "K." clear the title.
\AtBeginDocument{%
  \@ifundefined{cftappendicesnumwidth}{}{%
    \setlength{\cftappendicesnumwidth}{2.3em}%
  }%
}
\usepackage{amsmath}
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
\textwidth\paperwidth
\advance\textwidth -50mm
\oddsidemargin-1in
\advance\oddsidemargin 25mm
\evensidemargin-1in
\advance\evensidemargin 25mm
\topmargin -1in
\advance\topmargin 25mm
% No header content is ever displayed, but article.cls's default
% \headheight/\headsep still consume vertical space above \textheight unless
% zeroed -- previously pushing the printed top margin to ~37mm instead of the
% required 25mm. \footskip is sized to hold the 12pt footer page number
% within the 25mm bottom margin rather than beyond it.
\headheight 0pt
\headsep 0pt
\footskip 12mm
\textheight\paperheight
\advance\textheight -50mm
\advance\textheight -\footskip
\marginparwidth15mm
% WNE UW rule 8 requires a footnote wrapping to a second line to be indented
% "z odsadzeniem uwidaczniajacym numer przypisu" -- so the marker stays visible
% rather than being absorbed into the left margin. article.cls's \@makefntext
% indents only the first line and returns continuation lines flush left, which
% is the opposite. \hangindent indents every line after the first by the width
% reserved for the marker; the \hb@xt@ box holds the marker itself at that same
% width so the two line up. 1.5em matches the tab-width odsadzenie rule 14 uses
% for wrapped bibliography entries.
\long\def\@makefntext#1{%
  \noindent
  \hangindent 1.5em
  \hb@xt@ 1.5em{\@makefnmark\hfil}#1}
% Abstract environment: 12 pt body text at the document's own margins.
% The \null\vfil ... \vfil\null pair previously wrapping this body is the
% standard "vertically centre the abstract on its own page" idiom. It only
% works when the abstract is short enough to sit on one page with the glue
% around it. Once the text grew past that, the leading \vfil stretched to
% fill the whole page, stranding the "Abstract" heading alone on one page
% and pushing the body onto the next, itself indented from the top by the
% trailing \vfil. WNE UW rule 1c expects the abstract, Erasmus codes, and
% thematic classification to share a single page, so the abstract starts at
% the top of the page and flows normally instead.
% Body text stays at the document's 12 pt (WNE UW rule 2). The \large here
% previously rendered the abstract at 14.3 pt against 12 pt body text. Every
% font-size deviation the requirements allow is downward -- 10 pt for
% footnotes (rule 8), figure sources (rule 11), and long block quotations
% (rule 7) -- and none licenses a larger font, so there is no basis for
% setting the abstract above 12 pt.
% \quotation previously wrapped the body. Its \leftmargin/\rightmargin is
% \leftmargini -- 30 pt (10.5 mm) at 12 pt -- so the abstract printed at
% 35.5 mm on both sides while the Key words, Erasmus code, and thematic
% classification blocks directly beneath it on the same page sat at the
% required 25 mm. WNE UW rule 5 fixes all four margins at 25 mm and grants
% the abstract no exemption; it is not a block quotation (rule 7), which is
% the only body text the requirements indent. \quotation also forced a
% 1.5em first-line indent over the document's 1 cm (rule 4). Plain paragraph
% flow inherits both correctly.
% The faculty model thesis heads this page "Summary", not "Abstract".
\renewcommand{\abstractname}{Summary}
% ...and specifies the summary text as "justified text, no tab" ("tekst
% wyjustowany, bez tabulacji"). That overrides the document-wide 1 cm
% paragraph tab of rule 4 for this block alone: the abstract is the one place
% the model asks for unindented paragraphs. \setlength here is local, since a
% LaTeX environment is a group, so body text past \end{abstract} keeps its
% 1 cm indent.
\renewenvironment{abstract}{%
  \begin{center}
    \bfseries \abstractname
  \end{center}
  \setlength{\parindent}{0pt}%
}{%
}

\renewcommand\maketitle{%
  \begin{titlepage}%
    \let\footnotesize\small
    \let\footnoterule\relax
    \let \footnote \thanks
%   Annex D bolds exactly one element of the title page, the thesis title, and
%   labels everything in English. This page previously bolded the university,
%   the author and the degree line as well, and carried the Polish labels "Nr
%   albumu", "Praca magisterska" and "na kierunku" on an English thesis.
    \begin{center}%
      {\LARGE University of Warsaw\\
        Faculty of Economic Sciences\par}
      \vspace{1cm plus 1fill}
      {\Large\@author\par}
      \vspace{0.2cm}
      {\large Album N\textsuperscript{o}: \@nralbumu\par}
      \vspace{8mm plus .1fill}
      {\Huge\textbf{\@title}\par}
      \vspace{8mm plus .1fill}
      {\large \@rodzajpracy\ degree thesis\\[3pt]
        Field of the study: \@speciality \\
%----zakres---
        \@zakres \par}
      \vspace{2cm plus 1.5fill}
%   Annex D sets the supervisor block flush right over four lines: the lead-in,
%   the supervisor, their unit and "WNE UW". The tabular used here left-aligned
%   the block inside a right-hand box instead, so the lines did not share a
%   right edge. \raggedleft gives the true flush-right setting.
      {\large\raggedleft
        The thesis written under the supervision of\\
        \@opiekun\\
        \@jednostka
        WNE UW\par}
      \vspace{1cm plus 1fill}
      {\large \@date\par}
    \end{center}
    \@thanks
  \end{titlepage}%
  \begin{titlepage}
    \c@page=1
    \thispagestyle{plain}
    \fontsize{12pt}{14.4pt}\selectfont
    \null
    \vfill

%   Annex D gives each declaration one italic heading, its body, and a single
%   "Date ... Signature" line. This page previously carried the Polish
%   template's date and signature lines as well, un-removed, so each
%   declaration had two of them -- and the Polish ones printed *above* the
%   statement they belonged to, making "Data / Podpis opiekuna" the first text
%   on the page. A duplicate Polish heading, "Oswiadczenie autora (autorow)
%   pracy", sat above the English author statement for the same reason.
%   Headings are italic here, as in Annex D, not bold.
   \vspace{10mm}

   \textit{Declaration of the supervisor}
   \vspace{5mm}

   I hereby certify that the thesis submitted has been prepared under my supervision and I declare that it satisfies the requirements of submission in the
   proceedings for the award of a degree.

   \vspace{10mm}
   Date \hfill Signature of the Supervisor

   \vspace{25mm}
   \textit{Declaration of the author of the thesis}
   \vspace{5mm}

   Aware of legal liability I certify that the thesis submitted has been prepared by myself and does not include information gathered contrary to the law.

   I also declare that the thesis submitted has not been the subject of proceedings resulting in the award of a university degree.

   Furthermore I certify that the submitted version of the thesis is identical with its attached electronic version.

   \vspace{10mm}
   Date \hfill Signature of the Author

   \vspace{4cm}
  \end{titlepage}
  \setcounter{footnote}{0}%
  \setcounter{chapter}{1}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\nralbumu#1{\gdef\@nralbumu{#1}}
\def\@nralbumu{???\ClassError{pracamgr}{Brak numeru albumu}\@ehc}
\def\speciality#1{\gdef\@speciality{#1}}
\def\@speciality{???\ClassError{pracamgr}{Nie podano kierunku studiow}\@ehc}
%----zakres niekonieczny-----
\def\zakres#1{\gdef\@zakres{w zakresie \MakeUppercase{#1}}}
\def\@zakres{}
\def\opiekun#1{\gdef\@opiekun{#1}}
\def\@opiekun{???\ClassError{pracamgr}{Brak danych opiekuna pracy}\@ehc}
% Supervisor's unit, the third line of Annex D's title-page supervisor block
% ("from (name of the supervisor's unit)"). Optional: left empty the block
% prints the lead-in, the supervisor and "WNE UW" only. Set it with
% \jednostka{from the Department of ...} in the document preamble.
\def\jednostka#1{\gdef\@jednostka{#1\\}}
\def\@jednostka{}
\def\keywords#1{\gdef\@keywords{#1}}
\def\@keywords{???\ClassError{pracamgr}{Brak slow kluczowych}\@ehc}
\def\dziedzina#1{\gdef\@dziedzina{#1}}
\def\@dziedzina{???\ClassError{pracamgr}{Brak dziedziny Socrates-Erasmus}\@ehc}
\def\klasyfikacja#1{\gdef\@klasyfikacja{#1}}
\def\@klasyfikacja{???\ClassError{pracamgr}{Brak klasyfikacji tematycznej}\@ehc}
%-------------------nowe------------
\def\tytulang#1{\gdef\@tytulang{#1}}
\def\@tytulang{???\ClassError{pracamgr}{Brak tytulu po angielsku}\@ehc}


\newenvironment{summary}{%
  \titlepage
  \null\nobreak\vfil
  \@beginparpenalty\@lowpenalty
  \begin{center}%
    \bfseries\large Summary
    \@endparpenalty\@M
  \end{center}}%
{\par
  \vspace*{26pt}%
  \begin{center}%
    \bfseries\large Key words*
    \@endparpenalty\@M
  \end{center}
  \@keywords\par
  \vspace*{26pt}%
  \begin{center}%
    \bfseries\large Area of study (codes according to Erasmus Subject Area Codes List)
    \@endparpenalty\@M
  \end{center}
   \@dziedzina\par
  \vspace*{26pt}%
  \begin{center}%
    \bfseries\large Theme classification
    \@endparpenalty\@M
  \end{center}
  \@klasyfikacja\par
  \vspace*{26pt}%
%-------------------nowosc----------------
  \begin{center}%
    \bfseries\large The title of the thesis in Polish
    \@endparpenalty\@M
  \end{center}
  \@tytulang\par
  \nobreak\vfil\null\endtitlepage\cleardoublepage}

\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
    \hbox{}\thispagestyle{empty}\newpage\if@twocolumn\hbox{}\newpage\fi\fi\fi}

\renewcommand*\@seccntformat[1]{\csname the#1\endcsname.\enspace}
% Annex D, page 5: "Chapters should be written in capital letters according to
% the table of contents from the left margin. ... Font of thesis: Times New
% Roman, size 12 points. The same applies to the font of chapters and
% subsections."
%
% Two consequences the article defaults get wrong. Chapter headings printed in
% title case ("1. Introduction") where the model prints capitals, and the
% heading sizes stepped up with level -- 17.2 pt for chapters and 14.3 pt for
% sections against a 12 pt body -- where the model sets every heading at the
% body's 12 pt. Annex D's own "INTRODUCTION" measures 12.0 pt bold at the left
% margin.
%
% titlesec is used rather than patching \@startsection so the numbering format
% above stays in force: \thesection followed by a period and a space. Only the
% font and the case change. \MakeUppercase wraps the title text alone, so the
% number keeps its digits and the table of contents is uppercased separately
% below (uppercasing here would not reach the .toc file).
\RequirePackage{titlesec}
\titleformat{\section}{\normalsize\bfseries}{\thesection.}{0.5em}{\MakeUppercase}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection.}{0.5em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection.}{0.5em}{}
% Chapter entries in the table of contents carry the same capitals, so the body
% and the contents agree -- which is what "according to the table of contents"
% asks for.
% Annex D's contents runs dot leaders to the page number on every line,
% chapter entries included ("INTRODUCTION.........4",
% "CHAPTER I. <Title of the chapter>.....7"), and sets them all in regular
% weight. article.cls's \l@section instead sets chapter-level entries bold and
% closes with \hfil, so those lines had no leaders and did not match either
% Annex D or the sub-levels below them, which do. Redefined here with the same
% leader mechanism \l@chapter above already uses. \MakeUppercase carries the
% capitals over from the body headings, as "according to the table of
% contents" requires.
\renewcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\z@
    \addpenalty\@secpenalty
    \addvspace{1.0em \@plus\p@}%
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      \MakeUppercase{#1}\nobreak
      \leaders\hbox{$\m@th
        \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
      \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
    \endgroup
  \fi}
\def\numberline#1{\hb@xt@\@tempdima{#1.\hfil}}
\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\mdseries
      \leaders\hbox{$\m@th
        \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
      \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}

\endinput
