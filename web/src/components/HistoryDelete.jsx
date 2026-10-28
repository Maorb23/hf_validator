import React,{useState} from 'react';

export default function HistoryDelete({label,description,url,payload,action,busy=false,onDeleted}){
 const [confirming,setConfirming]=useState(false);
 async function remove(){try{await action(url,payload||{},'DELETE');setConfirming(false);onDeleted?.()}catch{}}
 return confirming?<div className="history-delete-confirm" role="group" aria-label={label+' confirmation'}><p>{description} Original execution artifacts remain on disk.</p><div className="form-actions"><button type="button" className="button danger secondary" disabled={busy} onClick={remove}>Confirm deletion</button><button type="button" className="button secondary" disabled={busy} onClick={()=>setConfirming(false)}>Keep history</button></div></div>:<button type="button" className="button danger secondary" disabled={busy} onClick={()=>setConfirming(true)}>{label}</button>;
}
