//! Low-level bytecode emitter, used by ast_builder.
//!
//! This API makes it easy to emit correct individual bytecode instructions.

// Most of this functionality isn't used yet.
#![allow(dead_code)]

use super::opcode::Opcode;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResumeKind {
    Normal = 0,
    Throw = 1,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AsyncFunctionResolveKind {
    Fulfill = 0,
    Reject = 1,
}

#[allow(non_camel_case_types)]
pub type u24 = u32;

/// Low-level bytecode emitter.
pub struct Emitter {
    bytecode: Vec<u8>,
    strings: Vec<String>,
}

/// The output of bytecode-compiling a script or module.
#[derive(Debug)]
pub struct EmitResult {
    pub bytecode: Vec<u8>,
    pub strings: Vec<String>,
}

impl Emitter {
    pub fn new() -> Self {
        Self {
            bytecode: Vec::new(),
            strings: Vec::new(),
        }
    }

    pub fn into_emit_result(self) -> EmitResult {
        EmitResult {
            bytecode: self.bytecode,
            strings: self.strings,
        }
    }

    fn emit1(&mut self, opcode: Opcode) {
        self.bytecode.push(opcode.to_byte());
    }

    pub fn emit_boolean(&mut self, value: bool) {
        self.emit1(if value { Opcode::True } else { Opcode::False });
    }

    pub fn emit_binary_op(&mut self, opcode: Opcode) {
        assert!(opcode.is_simple_binary_operator());
        self.emit1(opcode);
    }

    pub fn emit_unary_op(&mut self, opcode: Opcode) {
        assert!(opcode.is_simple_unary_operator());
        self.emit1(opcode);
    }

    fn emit_i8(&mut self, opcode: Opcode, value: i8) {
        self.emit_u8(opcode, value as u8);
    }

    fn emit_u8(&mut self, opcode: Opcode, value: u8) {
        self.bytecode.push(opcode.to_byte());
        self.bytecode.push(value);
    }

    fn emit_u16(&mut self, opcode: Opcode, value: u16) {
        self.bytecode.push(opcode.to_byte());
        self.bytecode.extend_from_slice(&value.to_le_bytes());
    }

    fn emit_u24(&mut self, opcode: Opcode, value: u24) {
        self.bytecode.push(opcode.to_byte());
        let slice = value.to_le_bytes();
        assert!(slice.len() == 4 && slice[3] == 0);
        self.bytecode.extend_from_slice(&slice[0..3]);
    }

    fn emit_i32(&mut self, opcode: Opcode, value: i32) {
        self.bytecode.push(opcode.to_byte());
        self.bytecode.extend_from_slice(&value.to_le_bytes());
    }

    fn emit_u32(&mut self, opcode: Opcode, value: u32) {
        self.bytecode.push(opcode.to_byte());
        self.bytecode.extend_from_slice(&value.to_le_bytes());
    }

    fn emit_with_name_index(&mut self, opcode: Opcode, name: &str) {
        self.bytecode.push(opcode.to_byte());
        self.emit_atom(name);
    }

    fn emit_aliased(&mut self, opcode: Opcode, hops: u8, slot: u24) {
        self.bytecode.push(opcode.to_byte());
        self.bytecode.push(hops);
        let slice = slot.to_le_bytes();
        assert!(slice.len() == 4 && slice[3] == 0);
        self.bytecode.extend_from_slice(&slice[0..3]);
    }

    fn emit_with_offset(&mut self, opcode: Opcode, offset: i32) {
        self.emit_i32(opcode, offset);
    }

    fn emit_atom(&mut self, value: &str) {
        let mut index = None;
        // Eventually we should be fancy and make this O(1)
        for (i, string) in self.strings.iter().enumerate() {
            if string == value {
                index = Some(i as u32);
            }
        }
        let index: u32 = match index {
            Some(index) => index,
            None => {
                let index = self.strings.len();
                self.strings.push(value.to_string());
                index as u32
            }
        };
        self.bytecode.extend_from_slice(&index.to_ne_bytes());
    }

    // Public methods to emit each instruction.

    pub fn nop(&mut self) {
        self.emit1(Opcode::Nop);
    }

    pub fn undefined(&mut self) {
        self.emit1(Opcode::Undefined);
    }

    pub fn get_rval(&mut self) {
        self.emit1(Opcode::GetRval);
    }

    pub fn enter_with(&mut self, static_with_index: u32) {
        self.emit_u32(Opcode::EnterWith, static_with_index);
    }

    pub fn leave_with(&mut self) {
        self.emit1(Opcode::LeaveWith);
    }

    pub fn return_(&mut self) {
        self.emit1(Opcode::Return);
    }

    pub fn goto(&mut self, offset: i32) {
        self.emit_with_offset(Opcode::Goto, offset);
    }

    pub fn if_eq(&mut self, offset: i32) {
        self.emit_with_offset(Opcode::IfEq, offset);
    }

    pub fn if_ne(&mut self, offset: i32) {
        self.emit_with_offset(Opcode::IfNe, offset);
    }

    pub fn arguments(&mut self) {
        self.emit1(Opcode::Arguments);
    }

    pub fn swap(&mut self) {
        self.emit1(Opcode::Swap);
    }

    pub fn pop_n(&mut self, n: u16) {
        self.emit_u16(Opcode::PopN, n);
    }

    pub fn dup(&mut self) {
        self.emit1(Opcode::Dup);
    }

    pub fn dup2(&mut self) {
        self.emit1(Opcode::Dup2);
    }

    // TODO - stronger typing of this parameter
    pub fn check_is_obj(&mut self, kind: u8) {
        self.emit_u8(Opcode::CheckIsObj, kind);
    }

    pub fn bit_or(&mut self) {
        self.emit1(Opcode::BitOr);
    }

    pub fn bit_xor(&mut self) {
        self.emit1(Opcode::BitXor);
    }

    pub fn bit_and(&mut self) {
        self.emit1(Opcode::BitAnd);
    }

    pub fn eq(&mut self) {
        self.emit1(Opcode::Eq);
    }

    pub fn ne(&mut self) {
        self.emit1(Opcode::Ne);
    }

    pub fn add(&mut self) {
        self.emit1(Opcode::Add);
    }

    pub fn sub(&mut self) {
        self.emit1(Opcode::Sub);
    }

    pub fn mul(&mut self) {
        self.emit1(Opcode::Mul);
    }

    pub fn div(&mut self) {
        self.emit1(Opcode::Div);
    }

    pub fn mod_(&mut self) {
        self.emit1(Opcode::Mod);
    }

    pub fn del_name(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::DelName, name);
    }

    pub fn del_prop(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::DelProp, name);
    }

    pub fn del_elem(&mut self) {
        self.emit1(Opcode::DelElem);
    }

    pub fn typeof_(&mut self) {
        self.emit1(Opcode::Typeof);
    }

    pub fn spread_call(&mut self) {
        self.emit1(Opcode::SpreadCall);
    }

    pub fn spread_new(&mut self) {
        self.emit1(Opcode::SpreadNew);
    }

    pub fn spread_eval(&mut self) {
        self.emit1(Opcode::SpreadEval);
    }

    pub fn dup_at(&mut self, n: u32) {
        self.emit_u24(Opcode::DupAt, n);
    }

    pub fn symbol(&mut self, symbol: u8) {
        self.emit_u8(Opcode::Symbol, symbol);
    }

    pub fn strict_del_prop(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::StrictDelProp, name);
    }

    pub fn strict_del_elem(&mut self) {
        self.emit1(Opcode::StrictDelElem);
    }

    pub fn strict_set_prop(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::StrictSetProp, name);
    }

    pub fn strict_set_name(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::StrictSetName, name);
    }

    pub fn strict_spread_eval(&mut self) {
        self.emit1(Opcode::StrictSpreadEval);
    }

    pub fn check_class_heritage(&mut self) {
        self.emit1(Opcode::CheckClassHeritage);
    }

    pub fn fun_with_proto(&mut self, func_index: u32) {
        self.emit_u32(Opcode::FunWithProto, func_index);
    }

    pub fn get_prop(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::GetProp, name);
    }

    pub fn set_prop(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::SetProp, name);
    }

    pub fn get_elem(&mut self) {
        self.emit1(Opcode::GetElem);
    }

    pub fn set_elem(&mut self) {
        self.emit1(Opcode::SetElem);
    }

    pub fn strict_set_elem(&mut self) {
        self.emit1(Opcode::StrictSetElem);
    }

    pub fn call(&mut self, argc: u16) {
        self.emit_u16(Opcode::Call, argc);
    }

    pub fn get_name(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::GetName, name);
    }

    pub fn double(&mut self, value: f64) {
        self.bytecode.push(Opcode::Double.to_byte());
        self.bytecode
            .extend_from_slice(&value.to_bits().to_le_bytes());
    }

    pub fn string(&mut self, value: &str) {
        self.bytecode.push(Opcode::String.to_byte());
        self.emit_atom(value);
    }

    pub fn zero(&mut self) {
        self.emit1(Opcode::Zero);
    }

    pub fn one(&mut self) {
        self.emit1(Opcode::One);
    }

    pub fn null(&mut self) {
        self.emit1(Opcode::Null);
    }

    pub fn is_constructing(&mut self) {
        self.emit1(Opcode::IsConstructing);
    }

    pub fn or(&mut self, offset: i32) {
        self.emit_with_offset(Opcode::Or, offset);
    }

    pub fn and(&mut self, offset: i32) {
        self.emit_with_offset(Opcode::And, offset);
    }

    pub fn table_switch(&mut self, _len: i32, _low: i32, _high: i32, _first_resume_index: u24) {
        unimplemented!();
    }

    pub fn strict_eq(&mut self) {
        self.emit1(Opcode::StrictEq);
    }

    pub fn strict_ne(&mut self) {
        self.emit1(Opcode::StrictNe);
    }

    pub fn throw_msg(&mut self, message_number: u16) {
        self.emit_u16(Opcode::ThrowMsg, message_number);
    }

    pub fn iter(&mut self) {
        self.emit1(Opcode::Iter);
    }

    pub fn more_iter(&mut self) {
        self.emit1(Opcode::MoreIter);
    }

    pub fn is_no_iter(&mut self) {
        self.emit1(Opcode::IsNoIter);
    }

    pub fn end_iter(&mut self) {
        self.emit1(Opcode::EndIter);
    }

    pub fn fun_apply(&mut self, argc: u16) {
        self.emit_u16(Opcode::FunApply, argc);
    }

    pub fn object(&mut self, object_index: u32) {
        self.emit_u32(Opcode::Object, object_index);
    }

    pub fn pop(&mut self) {
        self.emit1(Opcode::Pop);
    }

    pub fn new_(&mut self, argc: u16) {
        self.emit_u16(Opcode::New, argc);
    }

    pub fn obj_with_proto(&mut self) {
        self.emit1(Opcode::ObjWithProto);
    }

    pub fn get_arg(&mut self, arg_no: u16) {
        self.emit_u16(Opcode::GetArg, arg_no);
    }

    pub fn set_arg(&mut self, arg_no: u16) {
        self.emit_u16(Opcode::SetArg, arg_no);
    }

    pub fn get_local(&mut self, local_no: u24) {
        self.emit_u24(Opcode::GetLocal, local_no);
    }

    pub fn set_local(&mut self, local_no: u24) {
        self.emit_u24(Opcode::SetLocal, local_no);
    }

    pub fn uint16(&mut self, value: u16) {
        self.emit_u16(Opcode::Uint16, value);
    }

    pub fn new_init(&mut self, extra: u32) {
        self.emit_u32(Opcode::NewInit, extra);
    }

    pub fn new_array(&mut self, length: u32) {
        self.emit_u32(Opcode::NewArray, length);
    }

    pub fn new_object(&mut self, base_obj_index: u32) {
        self.emit_u32(Opcode::NewObject, base_obj_index);
    }

    pub fn init_home_object(&mut self) {
        self.emit1(Opcode::InitHomeObject);
    }

    pub fn init_prop(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::InitProp, name);
    }

    pub fn init_elem(&mut self) {
        self.emit1(Opcode::InitElem);
    }

    pub fn init_elem_inc(&mut self) {
        self.emit1(Opcode::InitElemInc);
    }

    pub fn init_elem_array(&mut self, index: u32) {
        self.emit_u32(Opcode::InitElemArray, index);
    }

    pub fn init_prop_getter(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::InitPropGetter, name);
    }

    pub fn init_prop_setter(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::InitPropSetter, name);
    }

    pub fn init_elem_getter(&mut self) {
        self.emit1(Opcode::InitElemGetter);
    }

    pub fn init_elem_setter(&mut self) {
        self.emit1(Opcode::InitElemSetter);
    }

    pub fn call_site_obj(&mut self, object_index: u32) {
        self.emit_u32(Opcode::CallSiteObj, object_index);
    }

    pub fn new_array_copy_on_write(&mut self, object_index: u32) {
        self.emit_u32(Opcode::NewArrayCopyOnWrite, object_index);
    }

    pub fn super_base(&mut self) {
        self.emit1(Opcode::SuperBase);
    }

    pub fn get_prop_super(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::GetPropSuper, name);
    }

    pub fn strict_set_prop_super(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::StrictSetPropSuper, name);
    }

    pub fn label(&mut self, offset: i32) {
        self.emit_with_offset(Opcode::Label, offset);
    }

    pub fn set_prop_super(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::SetPropSuper, name);
    }

    pub fn fun_call(&mut self, argc: u16) {
        self.emit_u16(Opcode::FunCall, argc);
    }

    pub fn loop_head(&mut self, ic_index: u32) {
        self.emit_u32(Opcode::LoopHead, ic_index);
    }

    pub fn bind_name(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::BindName, name);
    }

    pub fn set_name(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::SetName, name);
    }

    pub fn throw(&mut self) {
        self.emit1(Opcode::Throw);
    }

    pub fn in_(&mut self) {
        self.emit1(Opcode::In);
    }

    pub fn instanceof(&mut self) {
        self.emit1(Opcode::Instanceof);
    }

    pub fn debugger(&mut self) {
        self.emit1(Opcode::Debugger);
    }

    pub fn gosub(&mut self, offset: i32) {
        self.emit_with_offset(Opcode::Gosub, offset);
    }

    pub fn retsub(&mut self) {
        self.emit1(Opcode::Retsub);
    }

    pub fn exception(&mut self) {
        self.emit1(Opcode::Exception);
    }

    pub fn lineno(&mut self, lineno: u32) {
        self.emit_u32(Opcode::Lineno, lineno);
    }

    pub fn cond_switch(&mut self) {
        self.emit1(Opcode::CondSwitch);
    }

    pub fn case(&mut self, offset: i32) {
        self.emit_with_offset(Opcode::Case, offset);
    }

    pub fn default(&mut self, offset: i32) {
        self.emit_with_offset(Opcode::Default, offset);
    }

    pub fn eval(&mut self, argc: u16) {
        self.emit_u16(Opcode::Eval, argc);
    }

    pub fn strict_eval(&mut self, argc: u16) {
        self.emit_u16(Opcode::StrictEval, argc);
    }

    pub fn get_elem_super(&mut self) {
        self.emit1(Opcode::GetElemSuper);
    }

    pub fn resume_index(&mut self, resume_index: u24) {
        self.emit_u24(Opcode::ResumeIndex, resume_index);
    }

    pub fn def_fun(&mut self) {
        self.emit1(Opcode::DefFun);
    }

    pub fn def_const(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::DefConst, name);
    }

    pub fn def_var(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::DefVar, name);
    }

    pub fn lambda(&mut self, func_index: u32) {
        self.emit_u32(Opcode::Lambda, func_index);
    }

    pub fn lambda_arrow(&mut self, func_index: u32) {
        self.emit_u32(Opcode::LambdaArrow, func_index);
    }

    pub fn callee(&mut self) {
        self.emit1(Opcode::Callee);
    }

    pub fn pick(&mut self, n: u8) {
        self.emit_u8(Opcode::Pick, n);
    }

    pub fn try_(&mut self) {
        self.emit1(Opcode::Try);
    }

    pub fn finally(&mut self) {
        self.emit1(Opcode::Finally);
    }

    pub fn get_aliased_var(&mut self, hops: u8, slot: u24) {
        self.emit_aliased(Opcode::GetAliasedVar, hops, slot);
    }

    pub fn set_aliased_var(&mut self, hops: u8, slot: u24) {
        self.emit_aliased(Opcode::SetAliasedVar, hops, slot);
    }

    pub fn check_lexical(&mut self, local_no: u24) {
        self.emit_u24(Opcode::CheckLexical, local_no);
    }

    pub fn init_lexical(&mut self, local_no: u24) {
        self.emit_u24(Opcode::InitLexical, local_no);
    }

    pub fn check_aliased_lexical(&mut self, hops: u8, slot: u24) {
        self.emit_aliased(Opcode::CheckAliasedLexical, hops, slot);
    }

    pub fn init_aliased_lexical(&mut self, hops: u8, slot: u24) {
        self.emit_aliased(Opcode::InitAliasedLexical, hops, slot);
    }

    pub fn uninitialized(&mut self) {
        self.emit1(Opcode::Uninitialized);
    }

    pub fn get_intrinsic(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::GetIntrinsic, name);
    }

    pub fn set_intrinsic(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::SetIntrinsic, name);
    }

    pub fn call_iter(&mut self) {
        // JSOP_CALLITER has an operand in bytecode, for consistency with other
        // call opcodes, but it must be 0.
        self.emit_u16(Opcode::CallIter, 0);
    }

    pub fn init_locked_prop(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::InitLockedProp, name);
    }

    pub fn init_hidden_prop(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::InitHiddenProp, name);
    }

    pub fn new_target(&mut self) {
        self.emit1(Opcode::NewTarget);
    }

    pub fn pow(&mut self) {
        self.emit1(Opcode::Pow);
    }

    pub fn async_await(&mut self) {
        self.emit1(Opcode::AsyncAwait);
    }

    pub fn set_rval(&mut self) {
        self.emit1(Opcode::SetRval);
    }

    pub fn ret_rval(&mut self) {
        self.emit1(Opcode::RetRval);
    }

    pub fn get_gname(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::GetGname, name);
    }

    pub fn set_gname(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::SetGname, name);
    }

    pub fn strict_set_gname(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::StrictSetGname, name);
    }

    pub fn g_implicit_this(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::GImplicitThis, name);
    }

    pub fn set_elem_super(&mut self) {
        self.emit1(Opcode::SetElemSuper);
    }

    pub fn strict_set_elem_super(&mut self) {
        self.emit1(Opcode::StrictSetElemSuper);
    }

    pub fn reg_exp(&mut self, regexp_index: u32) {
        self.emit_u32(Opcode::RegExp, regexp_index);
    }

    pub fn init_g_lexical(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::InitGLexical, name);
    }

    pub fn def_let(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::DefLet, name);
    }

    pub fn check_obj_coercible(&mut self) {
        self.emit1(Opcode::CheckObjCoercible);
    }

    pub fn super_fun(&mut self) {
        self.emit1(Opcode::SuperFun);
    }

    pub fn super_call(&mut self, argc: u16) {
        self.emit_u16(Opcode::SuperCall, argc);
    }

    pub fn spread_super_call(&mut self) {
        self.emit1(Opcode::SpreadSuperCall);
    }

    pub fn class_constructor(&mut self, atom_index: u32) {
        self.emit_u32(Opcode::ClassConstructor, atom_index);
    }

    pub fn derived_constructor(&mut self, atom_index: u32) {
        self.emit_u32(Opcode::DerivedConstructor, atom_index);
    }

    pub fn throw_set_const(&mut self, local_no: u24) {
        self.emit_u24(Opcode::ThrowSetConst, local_no);
    }

    pub fn throw_set_aliased_const(&mut self, hops: u8, slot: u24) {
        self.emit_aliased(Opcode::ThrowSetAliasedConst, hops, slot);
    }

    pub fn init_hidden_prop_getter(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::InitHiddenPropGetter, name);
    }

    pub fn init_hidden_prop_setter(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::InitHiddenPropSetter, name);
    }

    pub fn init_hidden_elem_getter(&mut self) {
        self.emit1(Opcode::InitHiddenElemGetter);
    }

    pub fn init_hidden_elem_setter(&mut self) {
        self.emit1(Opcode::InitHiddenElemSetter);
    }

    pub fn init_hidden_elem(&mut self) {
        self.emit1(Opcode::InitHiddenElem);
    }

    pub fn get_import(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::GetImport, name);
    }

    pub fn debug_check_self_hosted(&mut self) {
        self.emit1(Opcode::DebugCheckSelfHosted);
    }

    pub fn optimize_spread_call(&mut self) {
        self.emit1(Opcode::OptimizeSpreadCall);
    }

    pub fn throw_set_callee(&mut self) {
        self.emit1(Opcode::ThrowSetCallee);
    }

    pub fn push_var_env(&mut self, scope_index: u32) {
        self.emit_u32(Opcode::PushVarEnv, scope_index);
    }

    pub fn pop_var_env(&mut self) {
        self.emit1(Opcode::PopVarEnv);
    }

    pub fn set_fun_name(&mut self, prefix_kind: u8) {
        self.emit_u8(Opcode::SetFunName, prefix_kind);
    }

    pub fn unpick(&mut self, n: u8) {
        self.emit_u8(Opcode::Unpick, n);
    }

    pub fn call_prop(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::CallProp, name);
    }

    pub fn function_this(&mut self) {
        self.emit1(Opcode::FunctionThis);
    }

    pub fn global_this(&mut self) {
        self.emit1(Opcode::GlobalThis);
    }

    pub fn is_gen_closing(&mut self) {
        self.emit1(Opcode::IsGenClosing);
    }

    pub fn uint24(&mut self, value: u24) {
        self.emit_u24(Opcode::Uint24, value);
    }

    pub fn check_this(&mut self) {
        self.emit1(Opcode::CheckThis);
    }

    pub fn check_return(&mut self) {
        self.emit1(Opcode::CheckReturn);
    }

    pub fn check_this_reinit(&mut self) {
        self.emit1(Opcode::CheckThisReinit);
    }

    pub fn async_resolve(&mut self, fulfill_or_reject: AsyncFunctionResolveKind) {
        self.emit_u8(Opcode::AsyncResolve, fulfill_or_reject as u8);
    }

    pub fn call_elem(&mut self) {
        self.emit1(Opcode::CallElem);
    }

    pub fn mutate_proto(&mut self) {
        self.emit1(Opcode::MutateProto);
    }

    pub fn get_bound_name(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::GetBoundName, name);
    }

    pub fn typeof_expr(&mut self) {
        self.emit1(Opcode::TypeofExpr);
    }

    pub fn freshen_lexical_env(&mut self) {
        self.emit1(Opcode::FreshenLexicalEnv);
    }

    pub fn recreate_lexical_env(&mut self) {
        self.emit1(Opcode::RecreateLexicalEnv);
    }

    pub fn push_lexical_env(&mut self, scope_index: u32) {
        self.emit_u32(Opcode::PushLexicalEnv, scope_index);
    }

    pub fn pop_lexical_env(&mut self) {
        self.emit1(Opcode::PopLexicalEnv);
    }

    pub fn debug_leave_lexical_env(&mut self) {
        self.emit1(Opcode::DebugLeaveLexicalEnv);
    }

    pub fn initial_yield(&mut self, resume_index: u24) {
        self.emit_u24(Opcode::InitialYield, resume_index);
    }

    pub fn yield_(&mut self, resume_index: u24) {
        self.emit_u24(Opcode::Yield, resume_index);
    }

    pub fn final_yield_rval(&mut self) {
        self.emit1(Opcode::FinalYieldRval);
    }

    pub fn resume(&mut self, kind: ResumeKind) {
        self.emit_u8(Opcode::Resume, kind as u8);
    }

    pub fn env_callee(&mut self, hops: u8) {
        self.emit_u8(Opcode::EnvCallee, hops);
    }

    pub fn force_interpreter(&mut self) {
        self.emit1(Opcode::ForceInterpreter);
    }

    pub fn after_yield(&mut self, ic_index: u32) {
        self.emit_u32(Opcode::AfterYield, ic_index);
    }

    pub fn await_(&mut self, resume_index: u24) {
        self.emit_u24(Opcode::Await, resume_index);
    }

    pub fn to_async_iter(&mut self) {
        self.emit1(Opcode::ToAsyncIter);
    }

    pub fn has_own(&mut self) {
        self.emit1(Opcode::HasOwn);
    }

    pub fn generator(&mut self) {
        self.emit1(Opcode::Generator);
    }

    pub fn bind_var(&mut self) {
        self.emit1(Opcode::BindVar);
    }

    pub fn bind_gname(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::BindGname, name);
    }

    pub fn int8(&mut self, value: i8) {
        self.emit_i8(Opcode::Int8, value);
    }

    pub fn int32(&mut self, value: i32) {
        self.emit_i32(Opcode::Int32, value);
    }

    pub fn length(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::Length, name);
    }

    pub fn hole(&mut self) {
        self.emit1(Opcode::Hole);
    }

    pub fn check_is_callable(&mut self, kind: u8) {
        self.emit_u8(Opcode::CheckIsCallable, kind);
    }

    pub fn try_destructuring(&mut self) {
        self.emit1(Opcode::TryDestructuring);
    }

    pub fn builtin_proto(&mut self, kind: u8) {
        self.emit_u8(Opcode::BuiltinProto, kind);
    }

    pub fn iter_next(&mut self) {
        self.emit1(Opcode::IterNext);
    }

    pub fn try_skip_await(&mut self) {
        self.emit1(Opcode::TrySkipAwait);
    }

    pub fn rest(&mut self) {
        self.emit1(Opcode::Rest);
    }

    pub fn toid(&mut self) {
        self.emit1(Opcode::Toid);
    }

    pub fn implicit_this(&mut self, name: &str) {
        self.emit_with_name_index(Opcode::ImplicitThis, name);
    }

    //pub fn loop_entry(&mut self, ic_index: u32, bits: u8) {
    //    self.emit_???(Opcode::LoopEntry, ic_index, bits);
    //}

    pub fn to_string(&mut self) {
        self.emit1(Opcode::ToString);
    }

    pub fn nop_destructuring(&mut self) {
        self.emit1(Opcode::NopDestructuring);
    }

    pub fn jump_target(&mut self, ic_index: u32) {
        self.emit_u32(Opcode::JumpTarget, ic_index);
    }

    pub fn call_ignores_rv(&mut self, argc: u16) {
        self.emit_u16(Opcode::CallIgnoresRv, argc);
    }

    pub fn import_meta(&mut self) {
        self.emit1(Opcode::ImportMeta);
    }

    pub fn dynamic_import(&mut self) {
        self.emit1(Opcode::DynamicImport);
    }

    pub fn inc(&mut self) {
        self.emit1(Opcode::Inc);
    }

    pub fn dec(&mut self) {
        self.emit1(Opcode::Dec);
    }

    pub fn to_numeric(&mut self) {
        self.emit1(Opcode::ToNumeric);
    }

    pub fn big_int(&mut self, const_index: u32) {
        self.emit_u32(Opcode::BigInt, const_index);
    }

    pub fn instrumentation_active(&mut self) {
        self.emit1(Opcode::InstrumentationActive);
    }

    pub fn instrumentation_callback(&mut self) {
        self.emit1(Opcode::InstrumentationCallback);
    }

    pub fn instrumentation_script_id(&mut self) {
        self.emit1(Opcode::InstrumentationScriptId);
    }
}
