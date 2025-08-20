"""
Internationalization and localization support for quantum task planning.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path
import gettext
from datetime import datetime


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    KOREAN = "ko"


class I18nManager:
    """
    Internationalization manager for quantum task planning.
    
    Provides multi-language support for all user-facing strings,
    quantum terminology, and cultural adaptations.
    """
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.quantum_terminology: Dict[str, Dict[str, str]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize translations
        self._load_translations()
        self._load_quantum_terminology()
        
        self.logger.info(f"I18n Manager initialized with default language: {default_language.value}")
    
    def _load_translations(self) -> None:
        """Load translation dictionaries for all supported languages."""
        
        # Core UI translations
        self.translations = {
            SupportedLanguage.ENGLISH.value: {
                # General UI
                "quantum_task_planner": "Quantum Task Planner",
                "create_task": "Create Task",
                "observe_task": "Observe Task", 
                "entangle_tasks": "Entangle Tasks",
                "execute_sequence": "Execute Sequence",
                "quantum_state": "Quantum State",
                "superposition": "Superposition",
                "entanglement": "Entanglement",
                "coherence": "Coherence",
                "interference": "Interference",
                
                # Task management
                "task_created": "Task created successfully",
                "task_observed": "Task observed and collapsed",
                "task_completed": "Task completed",
                "task_failed": "Task failed",
                "entanglement_created": "Quantum entanglement created",
                "bell_violation_detected": "Bell inequality violation detected",
                
                # Status messages
                "system_operational": "System operational",
                "quantum_coherence_maintained": "Quantum coherence maintained",
                "optimization_applied": "Performance optimization applied",
                "security_validation_passed": "Security validation passed",
                
                # Errors
                "error_task_not_found": "Task not found",
                "error_invalid_entanglement": "Invalid entanglement parameters",
                "error_coherence_lost": "Quantum coherence lost",
                "error_security_violation": "Security violation detected"
            },
            
            SupportedLanguage.SPANISH.value: {
                "quantum_task_planner": "Planificador de Tareas Cuánticas",
                "create_task": "Crear Tarea",
                "observe_task": "Observar Tarea",
                "entangle_tasks": "Entrelazar Tareas", 
                "execute_sequence": "Ejecutar Secuencia",
                "quantum_state": "Estado Cuántico",
                "superposition": "Superposición",
                "entanglement": "Entrelazamiento",
                "coherence": "Coherencia",
                "interference": "Interferencia",
                
                "task_created": "Tarea creada exitosamente",
                "task_observed": "Tarea observada y colapsada",
                "task_completed": "Tarea completada",
                "task_failed": "Tarea falló",
                "entanglement_created": "Entrelazamiento cuántico creado",
                "bell_violation_detected": "Violación de desigualdad de Bell detectada",
                
                "system_operational": "Sistema operacional",
                "quantum_coherence_maintained": "Coherencia cuántica mantenida",
                "optimization_applied": "Optimización de rendimiento aplicada",
                "security_validation_passed": "Validación de seguridad aprobada",
                
                "error_task_not_found": "Tarea no encontrada",
                "error_invalid_entanglement": "Parámetros de entrelazamiento inválidos",
                "error_coherence_lost": "Coherencia cuántica perdida",
                "error_security_violation": "Violación de seguridad detectada"
            },
            
            SupportedLanguage.FRENCH.value: {
                "quantum_task_planner": "Planificateur de Tâches Quantiques",
                "create_task": "Créer une Tâche",
                "observe_task": "Observer la Tâche",
                "entangle_tasks": "Intrication des Tâches",
                "execute_sequence": "Exécuter la Séquence",
                "quantum_state": "État Quantique",
                "superposition": "Superposition",
                "entanglement": "Intrication",
                "coherence": "Cohérence", 
                "interference": "Interférence",
                
                "task_created": "Tâche créée avec succès",
                "task_observed": "Tâche observée et effondrée",
                "task_completed": "Tâche terminée",
                "task_failed": "Échec de la tâche",
                "entanglement_created": "Intrication quantique créée",
                "bell_violation_detected": "Violation d'inégalité de Bell détectée",
                
                "system_operational": "Système opérationnel",
                "quantum_coherence_maintained": "Cohérence quantique maintenue",
                "optimization_applied": "Optimisation des performances appliquée",
                "security_validation_passed": "Validation de sécurité réussie",
                
                "error_task_not_found": "Tâche non trouvée",
                "error_invalid_entanglement": "Paramètres d'intrication invalides",
                "error_coherence_lost": "Cohérence quantique perdue",
                "error_security_violation": "Violation de sécurité détectée"
            },
            
            SupportedLanguage.GERMAN.value: {
                "quantum_task_planner": "Quantenaufgabenplaner",
                "create_task": "Aufgabe Erstellen",
                "observe_task": "Aufgabe Beobachten",
                "entangle_tasks": "Aufgaben Verschränken",
                "execute_sequence": "Sequenz Ausführen", 
                "quantum_state": "Quantenzustand",
                "superposition": "Überlagerung",
                "entanglement": "Verschränkung",
                "coherence": "Kohärenz",
                "interference": "Interferenz",
                
                "task_created": "Aufgabe erfolgreich erstellt",
                "task_observed": "Aufgabe beobachtet und kollabiert",
                "task_completed": "Aufgabe abgeschlossen",
                "task_failed": "Aufgabe fehlgeschlagen",
                "entanglement_created": "Quantenverschränkung erstellt",
                "bell_violation_detected": "Bell-Ungleichungsverletzung erkannt",
                
                "system_operational": "System betriebsbereit",
                "quantum_coherence_maintained": "Quantenkohärenz aufrechterhalten",
                "optimization_applied": "Leistungsoptimierung angewendet",
                "security_validation_passed": "Sicherheitsvalidierung bestanden",
                
                "error_task_not_found": "Aufgabe nicht gefunden",
                "error_invalid_entanglement": "Ungültige Verschränkungsparameter",
                "error_coherence_lost": "Quantenkohärenz verloren",
                "error_security_violation": "Sicherheitsverletzung erkannt"
            },
            
            SupportedLanguage.JAPANESE.value: {
                "quantum_task_planner": "量子タスクプランナー",
                "create_task": "タスクを作成",
                "observe_task": "タスクを観測",
                "entangle_tasks": "タスクをもつれさせる",
                "execute_sequence": "シーケンスを実行",
                "quantum_state": "量子状態",
                "superposition": "重ね合わせ",
                "entanglement": "もつれ",
                "coherence": "コヒーレンス",
                "interference": "干渉",
                
                "task_created": "タスクが正常に作成されました",
                "task_observed": "タスクが観測され、崩壊しました", 
                "task_completed": "タスクが完了しました",
                "task_failed": "タスクが失敗しました",
                "entanglement_created": "量子もつれが作成されました",
                "bell_violation_detected": "ベルの不等式違反が検出されました",
                
                "system_operational": "システムが稼働中",
                "quantum_coherence_maintained": "量子コヒーレンスが維持されています",
                "optimization_applied": "パフォーマンス最適化が適用されました",
                "security_validation_passed": "セキュリティ検証に合格しました",
                
                "error_task_not_found": "タスクが見つかりません",
                "error_invalid_entanglement": "無効なもつれパラメータ",
                "error_coherence_lost": "量子コヒーレンスが失われました",
                "error_security_violation": "セキュリティ違反が検出されました"
            },
            
            SupportedLanguage.CHINESE_SIMPLIFIED.value: {
                "quantum_task_planner": "量子任务规划器",
                "create_task": "创建任务",
                "observe_task": "观测任务", 
                "entangle_tasks": "纠缠任务",
                "execute_sequence": "执行序列",
                "quantum_state": "量子态",
                "superposition": "叠加态",
                "entanglement": "纠缠",
                "coherence": "相干性",
                "interference": "干涉",
                
                "task_created": "任务创建成功",
                "task_observed": "任务已观测并坍缩",
                "task_completed": "任务已完成",
                "task_failed": "任务失败",
                "entanglement_created": "量子纠缠已创建",
                "bell_violation_detected": "检测到贝尔不等式违反",
                
                "system_operational": "系统正常运行",
                "quantum_coherence_maintained": "量子相干性已维持",
                "optimization_applied": "性能优化已应用",
                "security_validation_passed": "安全验证已通过",
                
                "error_task_not_found": "未找到任务",
                "error_invalid_entanglement": "无效的纠缠参数",
                "error_coherence_lost": "量子相干性丢失",
                "error_security_violation": "检测到安全违规"
            }
        }
    
    def _load_quantum_terminology(self) -> None:
        """Load quantum physics terminology translations."""
        
        self.quantum_terminology = {
            SupportedLanguage.ENGLISH.value: {
                "wavefunction": "wavefunction",
                "probability_amplitude": "probability amplitude", 
                "quantum_measurement": "quantum measurement",
                "decoherence": "decoherence",
                "bell_inequality": "Bell inequality",
                "ghz_state": "GHZ state",
                "cluster_state": "cluster state",
                "von_neumann_entropy": "von Neumann entropy",
                "quantum_gate": "quantum gate",
                "hadamard_gate": "Hadamard gate",
                "pauli_gates": "Pauli gates",
                "quantum_circuit": "quantum circuit"
            },
            
            SupportedLanguage.SPANISH.value: {
                "wavefunction": "función de onda",
                "probability_amplitude": "amplitud de probabilidad",
                "quantum_measurement": "medición cuántica", 
                "decoherence": "decoherencia",
                "bell_inequality": "desigualdad de Bell",
                "ghz_state": "estado GHZ",
                "cluster_state": "estado cluster",
                "von_neumann_entropy": "entropía de von Neumann",
                "quantum_gate": "puerta cuántica",
                "hadamard_gate": "puerta Hadamard",
                "pauli_gates": "puertas Pauli",
                "quantum_circuit": "circuito cuántico"
            },
            
            SupportedLanguage.FRENCH.value: {
                "wavefunction": "fonction d'onde",
                "probability_amplitude": "amplitude de probabilité",
                "quantum_measurement": "mesure quantique",
                "decoherence": "décohérence", 
                "bell_inequality": "inégalité de Bell",
                "ghz_state": "état GHZ",
                "cluster_state": "état cluster",
                "von_neumann_entropy": "entropie de von Neumann",
                "quantum_gate": "porte quantique",
                "hadamard_gate": "porte Hadamard",
                "pauli_gates": "portes Pauli",
                "quantum_circuit": "circuit quantique"
            },
            
            SupportedLanguage.GERMAN.value: {
                "wavefunction": "Wellenfunktion",
                "probability_amplitude": "Wahrscheinlichkeitsamplitude",
                "quantum_measurement": "Quantenmessung",
                "decoherence": "Dekohärenz",
                "bell_inequality": "Bellsche Ungleichung",
                "ghz_state": "GHZ-Zustand",
                "cluster_state": "Cluster-Zustand", 
                "von_neumann_entropy": "von-Neumann-Entropie",
                "quantum_gate": "Quantengatter",
                "hadamard_gate": "Hadamard-Gatter",
                "pauli_gates": "Pauli-Gatter",
                "quantum_circuit": "Quantenschaltkreis"
            },
            
            SupportedLanguage.JAPANESE.value: {
                "wavefunction": "波動関数",
                "probability_amplitude": "確率振幅",
                "quantum_measurement": "量子測定",
                "decoherence": "デコヒーレンス",
                "bell_inequality": "ベルの不等式",
                "ghz_state": "GHZ状態",
                "cluster_state": "クラスター状態",
                "von_neumann_entropy": "フォンノイマンエントロピー",
                "quantum_gate": "量子ゲート",
                "hadamard_gate": "アダマールゲート",
                "pauli_gates": "パウリゲート",
                "quantum_circuit": "量子回路"
            },
            
            SupportedLanguage.CHINESE_SIMPLIFIED.value: {
                "wavefunction": "波函数",
                "probability_amplitude": "概率幅",
                "quantum_measurement": "量子测量",
                "decoherence": "失相干",
                "bell_inequality": "贝尔不等式",
                "ghz_state": "GHZ态",
                "cluster_state": "簇态",
                "von_neumann_entropy": "冯·诺伊曼熵",
                "quantum_gate": "量子门",
                "hadamard_gate": "阿达马门",
                "pauli_gates": "泡利门",
                "quantum_circuit": "量子电路"
            }
        }
    
    def set_language(self, language: SupportedLanguage) -> None:
        """Set the current language for translations."""
        self.current_language = language
        self.logger.info(f"Language set to: {language.value}")
    
    def translate(self, key: str, language: Optional[SupportedLanguage] = None) -> str:
        """Translate a UI string to the current or specified language."""
        lang = language or self.current_language
        
        translations = self.translations.get(lang.value, {})
        return translations.get(key, key)  # Return key if translation not found
    
    def translate_quantum_term(self, term: str, language: Optional[SupportedLanguage] = None) -> str:
        """Translate quantum physics terminology."""
        lang = language or self.current_language
        
        terminology = self.quantum_terminology.get(lang.value, {})
        return terminology.get(term, term)
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with their native names."""
        return [
            {"code": SupportedLanguage.ENGLISH.value, "name": "English", "native": "English"},
            {"code": SupportedLanguage.SPANISH.value, "name": "Spanish", "native": "Español"},
            {"code": SupportedLanguage.FRENCH.value, "name": "French", "native": "Français"},
            {"code": SupportedLanguage.GERMAN.value, "name": "German", "native": "Deutsch"},
            {"code": SupportedLanguage.JAPANESE.value, "name": "Japanese", "native": "日本語"},
            {"code": SupportedLanguage.CHINESE_SIMPLIFIED.value, "name": "Chinese (Simplified)", "native": "简体中文"},
            {"code": SupportedLanguage.PORTUGUESE.value, "name": "Portuguese", "native": "Português"},
            {"code": SupportedLanguage.ITALIAN.value, "name": "Italian", "native": "Italiano"},
            {"code": SupportedLanguage.RUSSIAN.value, "name": "Russian", "native": "Русский"},
            {"code": SupportedLanguage.KOREAN.value, "name": "Korean", "native": "한국어"}
        ]
    
    def format_datetime(self, dt: datetime, language: Optional[SupportedLanguage] = None) -> str:
        """Format datetime according to locale conventions."""
        lang = language or self.current_language
        
        # Define locale-specific datetime formats
        datetime_formats = {
            SupportedLanguage.ENGLISH.value: "%Y-%m-%d %H:%M:%S",
            SupportedLanguage.SPANISH.value: "%d/%m/%Y %H:%M:%S",
            SupportedLanguage.FRENCH.value: "%d/%m/%Y %H:%M:%S", 
            SupportedLanguage.GERMAN.value: "%d.%m.%Y %H:%M:%S",
            SupportedLanguage.JAPANESE.value: "%Y年%m月%d日 %H:%M:%S",
            SupportedLanguage.CHINESE_SIMPLIFIED.value: "%Y年%m月%d日 %H:%M:%S"
        }
        
        format_str = datetime_formats.get(lang.value, "%Y-%m-%d %H:%M:%S")
        return dt.strftime(format_str)
    
    def format_number(self, number: float, language: Optional[SupportedLanguage] = None) -> str:
        """Format numbers according to locale conventions."""
        lang = language or self.current_language
        
        # Simple number formatting based on locale
        if lang in [SupportedLanguage.GERMAN, SupportedLanguage.FRENCH]:
            # Use comma as decimal separator
            return f"{number:.3f}".replace(".", ",")
        else:
            # Use dot as decimal separator
            return f"{number:.3f}"
    
    def get_quantum_task_template(self, language: Optional[SupportedLanguage] = None) -> Dict[str, str]:
        """Get localized template for quantum task creation."""
        lang = language or self.current_language
        
        return {
            "title_placeholder": self.translate("enter_task_title", lang),
            "description_placeholder": self.translate("enter_task_description", lang),
            "priority_label": self.translate("priority", lang),
            "due_date_label": self.translate("due_date", lang),
            "tags_label": self.translate("tags", lang),
            "create_button": self.translate("create_task", lang),
            "cancel_button": self.translate("cancel", lang)
        }
    
    def localize_quantum_state_display(self, quantum_data: Dict[str, Any], language: Optional[SupportedLanguage] = None) -> Dict[str, str]:
        """Localize quantum state information for display."""
        lang = language or self.current_language
        
        localized = {}
        
        # Translate quantum state names
        state_translations = {
            "superposition": self.translate("superposition", lang),
            "collapsed": self.translate("collapsed", lang), 
            "entangled": self.translate("entangled", lang),
            "completed": self.translate("completed", lang),
            "failed": self.translate("failed", lang)
        }
        
        # Apply translations to quantum data
        for key, value in quantum_data.items():
            if key == "state" and value in state_translations:
                localized[key] = state_translations[value]
            elif key == "probability_amplitude":
                localized["probability_amplitude_label"] = self.translate_quantum_term("probability_amplitude", lang)
                localized["probability_amplitude_value"] = self.format_number(value, lang)
            elif key == "coherence_time":
                localized["coherence_label"] = self.translate_quantum_term("coherence", lang) 
                localized["coherence_value"] = self.format_number(value, lang)
            elif key == "entanglement_count":
                localized["entanglement_label"] = self.translate("entanglement", lang)
                localized["entanglement_count"] = str(value)
            elif isinstance(value, datetime):
                localized[key] = self.format_datetime(value, lang)
            elif isinstance(value, float):
                localized[key] = self.format_number(value, lang)
            else:
                localized[key] = str(value)
        
        return localized
    
    def get_error_message(self, error_key: str, context: Optional[Dict[str, Any]] = None, language: Optional[SupportedLanguage] = None) -> str:
        """Get localized error message with optional context substitution."""
        lang = language or self.current_language
        context = context or {}
        
        message = self.translate(f"error_{error_key}", lang)
        
        # Simple context substitution
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in message:
                message = message.replace(placeholder, str(value))
        
        return message
    
    def detect_user_language(self, user_context: Dict[str, Any]) -> SupportedLanguage:
        """Detect user's preferred language from context."""
        
        # Check explicit language preference
        if "language" in user_context:
            lang_code = user_context["language"]
            for lang in SupportedLanguage:
                if lang.value == lang_code:
                    return lang
        
        # Check Accept-Language header
        if "accept_language" in user_context:
            accept_lang = user_context["accept_language"].split(",")[0].split("-")[0]
            for lang in SupportedLanguage:
                if lang.value.startswith(accept_lang):
                    return lang
        
        # Check user location/timezone
        if "timezone" in user_context:
            timezone = user_context["timezone"]
            # Simple mapping based on common timezones
            timezone_to_lang = {
                "Europe/Madrid": SupportedLanguage.SPANISH,
                "Europe/Paris": SupportedLanguage.FRENCH,
                "Europe/Berlin": SupportedLanguage.GERMAN,
                "Asia/Tokyo": SupportedLanguage.JAPANESE,
                "Asia/Shanghai": SupportedLanguage.CHINESE_SIMPLIFIED
            }
            
            if timezone in timezone_to_lang:
                return timezone_to_lang[timezone]
        
        # Default fallback
        return self.default_language
    
    def export_translations(self, language: SupportedLanguage) -> Dict[str, Any]:
        """Export all translations for a language (for external tools)."""
        
        return {
            "language": language.value,
            "ui_translations": self.translations.get(language.value, {}),
            "quantum_terminology": self.quantum_terminology.get(language.value, {}),
            "export_timestamp": datetime.utcnow().isoformat()
        }
    
    def get_rtl_languages(self) -> List[SupportedLanguage]:
        """Get languages that require right-to-left text direction."""
        # None of the currently supported languages require RTL
        # But this could be extended for Arabic, Hebrew, etc.
        return []
    
    def get_cultural_preferences(self, language: SupportedLanguage) -> Dict[str, Any]:
        """Get cultural preferences for UI adaptations."""
        
        cultural_prefs = {
            SupportedLanguage.ENGLISH.value: {
                "date_format": "MM/DD/YYYY",
                "time_format": "12h",
                "number_format": "1,234.56",
                "currency_symbol": "$",
                "formal_address": False
            },
            SupportedLanguage.SPANISH.value: {
                "date_format": "DD/MM/YYYY", 
                "time_format": "24h",
                "number_format": "1.234,56",
                "currency_symbol": "€",
                "formal_address": True
            },
            SupportedLanguage.FRENCH.value: {
                "date_format": "DD/MM/YYYY",
                "time_format": "24h", 
                "number_format": "1 234,56",
                "currency_symbol": "€",
                "formal_address": True
            },
            SupportedLanguage.GERMAN.value: {
                "date_format": "DD.MM.YYYY",
                "time_format": "24h",
                "number_format": "1.234,56", 
                "currency_symbol": "€",
                "formal_address": True
            },
            SupportedLanguage.JAPANESE.value: {
                "date_format": "YYYY年MM月DD日",
                "time_format": "24h",
                "number_format": "1,234.56",
                "currency_symbol": "¥", 
                "formal_address": True
            },
            SupportedLanguage.CHINESE_SIMPLIFIED.value: {
                "date_format": "YYYY年MM月DD日",
                "time_format": "24h",
                "number_format": "1,234.56",
                "currency_symbol": "¥",
                "formal_address": True
            }
        }
        
        return cultural_prefs.get(language.value, cultural_prefs[SupportedLanguage.ENGLISH.value])
    
    def get_localization_stats(self) -> Dict[str, Any]:
        """Get localization statistics."""
        
        stats = {
            "supported_languages": len(SupportedLanguage),
            "current_language": self.current_language.value,
            "default_language": self.default_language.value,
            "total_translations": sum(len(trans) for trans in self.translations.values()),
            "quantum_terms": sum(len(terms) for terms in self.quantum_terminology.values()),
            "languages_with_full_translation": []
        }
        
        # Check translation completeness
        english_keys = set(self.translations.get(SupportedLanguage.ENGLISH.value, {}).keys())
        
        for lang_code, translations in self.translations.items():
            if lang_code == SupportedLanguage.ENGLISH.value:
                continue
            
            translation_keys = set(translations.keys())
            completeness = len(translation_keys & english_keys) / len(english_keys) if english_keys else 0
            
            if completeness >= 0.9:  # 90% or more translated
                stats["languages_with_full_translation"].append(lang_code)
        
        return stats