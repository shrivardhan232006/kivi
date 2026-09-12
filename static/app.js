/* Kivi Memory — Client-side JavaScript */

const API = '';  // Same origin

// --- Navigation ---
document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
        const view = btn.dataset.view;
        switchView(view);
    });
});

function switchView(viewName) {
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelector(`[data-view="${viewName}"]`).classList.add('active');
    document.getElementById(`view-${viewName}`).classList.add('active');

    // Load data for the view
    if (viewName === 'memories') loadMemories();
    if (viewName === 'history') loadHistory();
    if (viewName === 'dictionary') loadDictionary();
}

// --- Chat ---
let chatHistory = [];

const chatInput = document.getElementById('chat-input');
const followupInput = document.getElementById('chat-followup-input');
const heroGrid = document.getElementById('hero-grid');
const chatThreadContainer = document.getElementById('chat-thread-container');

if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });
}

if (followupInput) {
    followupInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendFollowupQuestion();
        }
    });
}

// Global Cmd+K / Ctrl+K focus
window.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        switchView('chat');
        if (chatThreadContainer && !chatThreadContainer.classList.contains('hidden')) {
            if (followupInput) {
                followupInput.focus();
                followupInput.select();
            }
        } else if (chatInput) {
            chatInput.focus();
            chatInput.select();
        }
    }
});

function switchToChatMode() {
    if (heroGrid) heroGrid.classList.add('hidden');
    if (chatThreadContainer) chatThreadContainer.classList.remove('hidden');
}

function switchToHeroMode() {
    if (heroGrid) heroGrid.classList.remove('hidden');
    if (chatThreadContainer) chatThreadContainer.classList.add('hidden');
}

function askQuestion(q) {
    sendQuestion(q);
}

function sendFollowupQuestion() {
    if (!followupInput) return;
    const q = followupInput.value.trim();
    if (!q) return;
    followupInput.value = '';
    sendQuestion(q);
}

function clearChat() {
    chatHistory = [];
    fetch(`${API}/api/clear-chat`, { method: 'POST' }).catch(() => {});
    const messagesDiv = document.getElementById('chat-messages');
    if (messagesDiv) messagesDiv.innerHTML = '';
    if (chatInput) {
        chatInput.value = '';
        chatInput.placeholder = 'Find the dictation I did around 5 PM in Slack and polish it for the meeting.';
    }
    if (followupInput) {
        followupInput.value = '';
    }
    switchToHeroMode();
}

async function sendQuestion(explicitQuestion = null) {
    let question = explicitQuestion;
    if (!question) {
        if (followupInput && followupInput.value.trim()) {
            question = followupInput.value.trim();
            followupInput.value = '';
        } else if (chatInput && chatInput.value.trim()) {
            question = chatInput.value.trim();
            chatInput.value = '';
        } else if (chatInput && chatInput.placeholder) {
            question = chatInput.placeholder;
        }
    }

    if (!question) return;

    // Shift to chat interface!
    switchToChatMode();

    const messagesDiv = document.getElementById('chat-messages');
    if (!messagesDiv) return;

    // Add user message
    messagesDiv.innerHTML += `
        <div class="chat-msg user">
            <div class="msg-bubble">${escapeHtml(question)}</div>
        </div>`;

    // Add loading indicator
    const loadingId = 'loading-' + Date.now();
    messagesDiv.innerHTML += `
        <div class="chat-msg assistant" id="${loadingId}">
            <div class="msg-bubble msg-loading">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>
        </div>`;

    scrollChat();

    const historyToSend = [...chatHistory];
    chatHistory.push({ role: 'user', content: question });

    try {
        const resp = await fetch(`${API}/api/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, history: historyToSend }),
        });

        const data = await resp.json();
        const loading = document.getElementById(loadingId);
        if (loading) loading.remove();

        if (data.answer) {
            chatHistory.push({ role: 'assistant', content: data.answer });
            if (chatHistory.length > 8) {
                chatHistory = chatHistory.slice(-8);
            }
        }

        // Build response HTML
        let metaHtml = '';
        if (data.tool_name) {
            metaHtml += `<span class="meta-badge badge-tool">${data.tool_name}</span>`;
        }
        if (data.fast_path) {
            metaHtml += `<span class="meta-badge badge-fast">fast-path</span>`;
        }
        if (data.verification?.passed) {
            metaHtml += `<span class="meta-badge badge-verified">verified</span>`;
        } else if (data.verification?.flagged_claims?.length) {
            metaHtml += `<span class="meta-badge badge-flagged">partial</span>`;
        }
        metaHtml += `<span>${data.latency_ms?.toFixed(0)}ms</span>`;
        metaHtml += `<span>${data.model_calls} LLM calls</span>`;

        if (data.tool_name === 'update_dictionary') {
            metaHtml += `<span class="meta-badge badge-dict">dictionary updated</span>`;
            if (data.tool_args?.raw_form) {
                metaHtml += `<span>${escapeHtml(data.tool_args.raw_form)} &rarr; ${escapeHtml(data.tool_args.corrected_form)}</span>`;
            }
            loadChatDictionary();
        }
        metaHtml += `<button class="dict-quick-btn" onclick="openDictionaryWithPrefill('${escapeHtml(data.tool_args?.raw_form || '')}', '${escapeHtml(data.tool_args?.corrected_form || '')}')">✎ dictionary</button>`;

        let sourcesHtml = '';
        if (data.source_record_ids?.length) {
            sourcesHtml = `<div class="msg-sources">`;
            for (const id of data.source_record_ids.slice(0, 6)) {
                sourcesHtml += `<a class="source-link" onclick="showHistoryRecord('${id}')">${id}</a>`;
            }
            if (data.source_record_ids.length > 6) {
                sourcesHtml += `<span class="source-link">+${data.source_record_ids.length - 6} more</span>`;
            }
            sourcesHtml += `</div>`;
        }

        messagesDiv.innerHTML += `
            <div class="chat-msg assistant">
                <div class="msg-bubble">${formatAnswer(data.answer)}</div>
                <div class="msg-meta">${metaHtml}</div>
                ${sourcesHtml}
            </div>`;

    } catch (err) {
        const loading = document.getElementById(loadingId);
        if (loading) loading.remove();
        messagesDiv.innerHTML += `
            <div class="chat-msg assistant">
                <div class="msg-bubble" style="color: var(--danger);">Error: ${escapeHtml(err.message)}</div>
            </div>`;
    }

    scrollChat();
    if (followupInput) {
        followupInput.focus();
    }
}

function scrollChat() {
    const followupCard = document.getElementById('chat-followup-card');
    if (followupCard) {
        followupCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else {
        const messages = document.getElementById('chat-messages');
        if (messages && messages.lastElementChild) {
            messages.lastElementChild.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
}

function formatAnswer(text) {
    if (!text) return '<em style="color: var(--text-muted);">No answer</em>';
    return escapeHtml(text).replace(/\n/g, '<br>');
}

// --- Memories ---
let currentMemoryFilter = 'all';

function filterMemories(kind) {
    currentMemoryFilter = kind;
    document.querySelectorAll('.filter-chip[data-kind]').forEach(c => c.classList.remove('active'));
    document.querySelector(`.filter-chip[data-kind="${kind}"]`).classList.add('active');
    loadMemories();
}

async function loadMemories() {
    const list = document.getElementById('memories-list');
    const kindParam = currentMemoryFilter !== 'all' ? `?kind=${currentMemoryFilter}` : '';

    try {
        const resp = await fetch(`${API}/api/memories${kindParam}`);
        const data = await resp.json();

        if (!data.groups?.length) {
            list.innerHTML = '<p style="color: var(--text-muted); padding: 40px; text-align: center;">No memories found. Run the ingestion pipeline first.</p>';
            return;
        }

        list.innerHTML = data.groups.map((group, idx) => {
            const itemsHtml = group.items.map(item => `
                <div class="memory-item">
                    <div class="mem-content">
                        <div class="mem-predicate">${escapeHtml(item.predicate)}</div>
                        <div class="mem-value">${escapeHtml(item.value)}</div>
                        ${item.qualifier ? `<div class="mem-qualifier">${escapeHtml(item.qualifier)}</div>` : ''}
                    </div>
                    <span class="mem-kind-badge kind-${item.kind}">${item.kind}</span>
                    <button class="mem-delete" onclick="deleteMemory('${item.id}')" title="Delete">&#10005;</button>
                </div>
            `).join('');

            const kindsStr = group.kinds.join(', ');
            const dateStr = new Date(group.last_updated).toLocaleDateString();

            return `
                <div class="memory-group">
                    <div class="memory-group-header" onclick="toggleGroup(${idx})">
                        <div class="group-info">
                            <span class="group-subject">${escapeHtml(group.subject)}</span>
                            <span class="group-count">${group.count} memories</span>
                        </div>
                        <div class="group-meta">
                            <span>${kindsStr}</span>
                            <span>${dateStr}</span>
                        </div>
                    </div>
                    <div class="memory-group-items" id="mem-group-${idx}">
                        ${itemsHtml}
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        list.innerHTML = `<p style="color: var(--danger); padding: 20px;">Error loading memories: ${err.message}</p>`;
    }
}

function toggleGroup(idx) {
    const el = document.getElementById(`mem-group-${idx}`);
    el.classList.toggle('expanded');
}

async function deleteMemory(id) {
    if (!confirm('Delete this memory?')) return;
    try {
        await fetch(`${API}/api/memories/${id}`, { method: 'DELETE' });
        loadMemories();
    } catch (err) {
        alert('Error deleting: ' + err.message);
    }
}

// --- History ---
let historyPage = 1;

document.getElementById('history-search').addEventListener('input', debounce(() => {
    historyPage = 1;
    loadHistory();
}, 300));

async function loadHistory() {
    const list = document.getElementById('history-list');
    const search = document.getElementById('history-search').value;
    const appFilter = document.getElementById('history-app-filter').value;

    const params = new URLSearchParams({ page: historyPage, per_page: 50 });
    if (search) params.set('search', search);
    if (appFilter) params.set('app_filter', appFilter);

    try {
        const resp = await fetch(`${API}/api/history?${params}`);
        const data = await resp.json();

        // Populate app filter if needed
        const select = document.getElementById('history-app-filter');
        if (select.options.length <= 1) {
            const statsResp = await fetch(`${API}/api/stats`);
            const stats = await statsResp.json();
            stats.apps?.forEach(app => {
                const opt = document.createElement('option');
                opt.value = app;
                opt.textContent = app;
                select.appendChild(opt);
            });
        }

        if (!data.records?.length) {
            list.innerHTML = '<p style="color: var(--text-muted); padding: 40px; text-align: center;">No history records found.</p>';
            document.getElementById('history-pagination').innerHTML = '';
            return;
        }

        list.innerHTML = data.records.map(rec => {
            const time = new Date(rec.timestamp).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
            const date = new Date(rec.timestamp).toLocaleDateString();
            return `
                <div class="history-item" onclick="showHistoryRecord('${rec.id}')">
                    <span class="history-app">${escapeHtml(rec.app)}</span>
                    <span class="history-text">${escapeHtml(rec.canonical_text)}</span>
                    <span class="history-time">${date} ${time}</span>
                </div>
            `;
        }).join('');

        // Pagination
        const pagDiv = document.getElementById('history-pagination');
        if (data.pages > 1) {
            let pagHtml = '';
            for (let i = 1; i <= data.pages; i++) {
                pagHtml += `<button class="page-btn ${i === data.page ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
            }
            pagDiv.innerHTML = pagHtml;
        } else {
            pagDiv.innerHTML = '';
        }
    } catch (err) {
        list.innerHTML = `<p style="color: var(--danger); padding: 20px;">Error: ${err.message}</p>`;
    }
}

function goToPage(p) {
    historyPage = p;
    loadHistory();
}

async function showHistoryRecord(id) {
    try {
        const resp = await fetch(`${API}/api/history/${id}`);
        const rec = await resp.json();
        const derived = rec.derived_memories?.length
            ? rec.derived_memories.map(m => `  - [${m.kind}] ${m.subject}: ${m.predicate} = ${m.value}`).join('\n')
            : '  (none)';
        alert(`Record: ${id}\nApp: ${rec.app}\nTime: ${rec.timestamp}\n\nRaw: ${rec.raw_text}\n\nFormatted: ${rec.formatted_text}\n\nCanonical: ${rec.canonical_text}\n\nDerived memories:\n${derived}`);
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

// --- Dictionary ---
async function loadDictionary() {
    const list = document.getElementById('dictionary-list');
    const sugSection = document.getElementById('suggestions-section');

    try {
        const [dictResp, sugResp] = await Promise.all([
            fetch(`${API}/api/dictionary`),
            fetch(`${API}/api/dictionary/suggestions`),
        ]);
        const dictData = await dictResp.json();
        const sugData = await sugResp.json();

        // Suggestions
        if (sugData.suggestions?.length) {
            sugSection.innerHTML = `
                <div class="suggestion-banner">
                    <div class="suggestion-title">Suggested additions (${sugData.suggestions.length})</div>
                    ${sugData.suggestions.map(s => `
                        <div class="suggestion-item">
                            <span class="suggestion-text">"${escapeHtml(s.raw_form)}" -> ${escapeHtml(s.corrected_form)}</span>
                            <span class="suggestion-count">${s.occurrence_count}x</span>
                            <div class="suggestion-actions">
                                <button class="accept" onclick="handleSuggestion(${s.id}, 'accept')">add</button>
                                <button class="dismiss" onclick="handleSuggestion(${s.id}, 'dismiss')">skip</button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        } else {
            sugSection.innerHTML = '';
        }

        // Entries
        if (!dictData.entries?.length) {
            list.innerHTML = '<p style="color: var(--text-muted); padding: 20px; text-align: center;">No dictionary entries yet.</p>';
            return;
        }

        list.innerHTML = dictData.entries.map(e => `
            <div class="dict-entry">
                <span class="dict-raw">"${escapeHtml(e.raw_form)}"</span>
                <span class="dict-arrow-icon" style="color: var(--text-muted);">&#8594;</span>
                <span class="dict-corrected">${escapeHtml(e.corrected_form)}</span>
            </div>
        `).join('');
    } catch (err) {
        list.innerHTML = `<p style="color: var(--danger); padding: 20px;">Error: ${err.message}</p>`;
    }
}

async function addDictEntry() {
    const raw = document.getElementById('dict-raw').value.trim();
    const corrected = document.getElementById('dict-corrected').value.trim();
    if (!raw || !corrected) return;

    try {
        await fetch(`${API}/api/dictionary`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ raw_form: raw, corrected_form: corrected }),
        });
        document.getElementById('dict-raw').value = '';
        document.getElementById('dict-corrected').value = '';
        loadDictionary();
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

async function handleSuggestion(id, action) {
    try {
        await fetch(`${API}/api/dictionary/suggestions/${id}/${action}`, { method: 'POST' });
        loadDictionary();
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

// --- In-Chat Dictionary Drawer ---
async function loadChatDictionary() {
    const container = document.getElementById('chat-dict-entries');
    if (!container) return;

    try {
        const resp = await fetch(`${API}/api/dictionary`);
        const data = await resp.json();
        if (!data.entries?.length) {
            container.innerHTML = '<span style="color: var(--text-muted); font-size: 12px;">No dictionary entries yet.</span>';
            return;
        }

        container.innerHTML = data.entries.map(e => `
            <div class="dict-panel-chip">
                <span class="dict-chip-raw">${escapeHtml(e.raw_form)}</span>
                <span class="dict-chip-arrow">&rarr;</span>
                <span class="dict-chip-corrected">${escapeHtml(e.corrected_form)}</span>
                <button class="dict-chip-del" onclick="deleteChatDictEntry('${escapeHtml(e.raw_form)}')" title="Delete">&times;</button>
            </div>
        `).join('');
    } catch (err) {
        // Non-critical
    }
}

function toggleChatDictionary(forceOpen = null) {
    const panel = document.getElementById('chat-dict-panel');
    const btn = document.getElementById('btn-dict-toggle');
    if (!panel) return;

    const shouldOpen = forceOpen !== null ? forceOpen : panel.classList.contains('hidden');
    if (shouldOpen) {
        panel.classList.remove('hidden');
        if (btn) btn.classList.add('active');
        loadChatDictionary();
        const rawInput = document.getElementById('chat-dict-raw');
        if (rawInput) rawInput.focus();
    } else {
        panel.classList.add('hidden');
        if (btn) btn.classList.remove('active');
    }
}

function openDictionaryWithPrefill(raw = '', corrected = '') {
    toggleChatDictionary(true);
    const rawInput = document.getElementById('chat-dict-raw');
    const correctedInput = document.getElementById('chat-dict-corrected');
    if (rawInput) rawInput.value = raw;
    if (correctedInput) correctedInput.value = corrected;
    if (rawInput && !raw) rawInput.focus();
    else if (correctedInput) correctedInput.focus();
}

async function addChatDictEntry() {
    const rawInput = document.getElementById('chat-dict-raw');
    const correctedInput = document.getElementById('chat-dict-corrected');
    if (!rawInput || !correctedInput) return;

    const raw_form = rawInput.value.trim();
    const corrected_form = correctedInput.value.trim();
    if (!raw_form || !corrected_form) {
        alert('Please provide both the spoken form and the corrected form.');
        return;
    }

    try {
        await fetch(`${API}/api/dictionary`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ raw_form, corrected_form }),
        });
        rawInput.value = '';
        correctedInput.value = '';
        loadChatDictionary();
        loadDictionary();
    } catch (err) {
        alert('Error adding dictionary entry: ' + err.message);
    }
}

async function deleteChatDictEntry(raw_form) {
    try {
        await fetch(`${API}/api/dictionary/${encodeURIComponent(raw_form)}`, {
            method: 'DELETE',
        });
        loadChatDictionary();
        loadDictionary();
    } catch (err) {
        alert('Error deleting entry: ' + err.message);
    }
}

// --- Stats ---
async function loadStats() {
    try {
        const resp = await fetch(`${API}/api/stats`);
        const stats = await resp.json();
        
        const memCount = document.getElementById('stat-memories-count');
        if (memCount) {
            memCount.textContent = stats.memory_valid_count || 68;
        }

        const histCount = document.getElementById('stat-history-count');
        if (histCount) {
            histCount.textContent = stats.history_count || 500;
        }
    } catch (err) {
        // Stats loading is non-critical
    }
}

// --- Utilities ---
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function debounce(fn, ms) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
    };
}

// --- Init ---
loadStats();
